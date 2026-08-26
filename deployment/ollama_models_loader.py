"""
Загрузка modules/*/ollama_models.yaml для setup-full pull моделей Ollama.

Только модули с зависимостью ollama_framework (requires/extends) или сам
ollama_framework. Discovery через ModuleCatalog, без Django.

CLI: python ollama_models_loader.py --root PATH --json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from cli_locale import t  # noqa: E402
from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from env_file_loader import load_project_env, parse_env_file  # noqa: E402
from lifecycle.modules.catalog import ModuleCatalog  # noqa: E402

from modules.ollama_framework.deployment.model_names import (  # noqa: E402
    is_ollama_library_name,
)

OLLAMA_MODELS_FILENAME = 'ollama_models.yaml'
INTEGRATIONS_FILENAME = 'integrations.yaml'
OLLAMA_FRAMEWORK_MODULE = 'ollama_framework'


@dataclass(frozen=True)
class ResolvedOllamaModel:
    name: str
    kind: str = ''
    required: bool = True
    source_module: str = ''
    source_file: str = ''


@dataclass
class OllamaModelsAggregate:
    models: list[ResolvedOllamaModel] = field(default_factory=list)
    modules: list[str] = field(default_factory=list)
    skipped_modules: list[str] = field(default_factory=list)


def _warn(message: str) -> None:
    print(format_console('warning', message), file=sys.stderr)


def _as_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        return [text] if text else []
    if isinstance(value, (list, tuple)):
        items: list[str] = []
        for item in value:
            if not isinstance(item, str):
                continue
            text = item.strip()
            if text:
                items.append(text)
        return items
    return []


def load_merged_env(root: Path) -> dict[str, str]:
    """Корневой .env + env/*.env + modules/**/.env (позже перекрывает раньше)."""
    merged = load_project_env(root)
    modules_dir = root / 'modules'
    if not modules_dir.is_dir():
        return merged

    env_files: list[Path] = []
    for env_file in modules_dir.rglob('.env'):
        if env_file.is_file() and not env_file.name.endswith('.example'):
            env_files.append(env_file)

    for env_file in sorted(env_files):
        merged.update(parse_env_file(env_file))
    return merged


def _parse_integrations(path: Path) -> tuple[str, ...]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError:
        return ()
    if not text.strip():
        return ()
    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError:
        return ()
    if not isinstance(data, dict):
        return ()
    requires = _as_str_list(data.get('requires'))
    extends = _as_str_list(data.get('extends'))
    return tuple(dict.fromkeys([*requires, *extends]))


def _module_may_declare_models(module_dir: Path) -> bool:
    if module_dir.name == OLLAMA_FRAMEWORK_MODULE:
        return True
    integrations_path = module_dir / INTEGRATIONS_FILENAME
    if not integrations_path.is_file():
        return False
    peers = _parse_integrations(integrations_path)
    return OLLAMA_FRAMEWORK_MODULE in peers


def _resolve_model_name(
    entry: dict[str, Any],
    *,
    env_values: dict[str, str],
) -> str:
    explicit = str(entry.get('name') or '').strip()
    if explicit:
        return explicit

    env_key = str(entry.get('env') or '').strip()
    if env_key:
        value = (env_values.get(env_key) or '').strip()
        if value:
            return value
        default = str(entry.get('default') or '').strip()
        if default:
            return default
        return ''

    return ''


def _parse_models_file(
    path: Path,
    *,
    module_dir_name: str,
    env_values: dict[str, str],
) -> list[ResolvedOllamaModel]:
    try:
        text = path.read_text(encoding='utf-8')
    except OSError as exc:
        _warn(t('yaml_read_failed', path=path, exc=exc))
        return []

    if not text.strip():
        _warn(t('yaml_file_empty', path=path))
        return []

    try:
        data = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        _warn(t('yaml_parse_error', path=path, exc=exc))
        return []

    if not isinstance(data, dict):
        _warn(t('yaml_root_must_be_object', path=path))
        return []

    module = str(data.get('module') or module_dir_name).strip() or module_dir_name
    raw_models = data.get('models')
    if raw_models is None:
        return []
    if not isinstance(raw_models, list):
        _warn(f'{path}: models должен быть списком')
        return []

    resolved: list[ResolvedOllamaModel] = []
    relative = path.as_posix()
    for index, item in enumerate(raw_models):
        if not isinstance(item, dict):
            _warn(f'{path}: models[{index}] должен быть объектом')
            continue

        name = _resolve_model_name(item, env_values=env_values)
        if not name:
            _warn(f'{path}: models[{index}] — пустое имя (нужны name или env+default)')
            continue
        if not is_ollama_library_name(name):
            fallback = str(item.get('default') or '').strip()
            if fallback and is_ollama_library_name(fallback):
                _warn(
                    f'{path}: {name} — снимок Hugging Face, для Ollama берём {fallback}'
                )
                name = fallback
            else:
                _warn(f'{path}: {name} — не имя библиотеки Ollama, пропуск')
                continue

        kind = str(item.get('kind') or '').strip().lower()
        required_raw = item.get('required', True)
        required = bool(required_raw) if required_raw is not None else True

        resolved.append(
            ResolvedOllamaModel(
                name=name,
                kind=kind,
                required=required,
                source_module=module,
                source_file=relative,
            )
        )
    return resolved


def _dedupe_models(models: list[ResolvedOllamaModel]) -> list[ResolvedOllamaModel]:
    by_name: dict[str, ResolvedOllamaModel] = {}
    for model in models:
        key = model.name.strip().lower()
        existing = by_name.get(key)
        if existing is None:
            by_name[key] = model
            continue
        if model.required and not existing.required:
            by_name[key] = ResolvedOllamaModel(
                name=model.name,
                kind=model.kind or existing.kind,
                required=True,
                source_module=model.source_module,
                source_file=model.source_file,
            )
    return list(by_name.values())


@lru_cache(maxsize=8)
def load_resolved_models(project_root: str) -> tuple[ResolvedOllamaModel, ...]:
    root = Path(project_root).resolve()
    catalog = ModuleCatalog.from_env(root)
    env_values = load_merged_env(root)
    collected: list[ResolvedOllamaModel] = []

    for module_dir in catalog.iter_module_dirs():
        path = module_dir / OLLAMA_MODELS_FILENAME
        if not path.is_file():
            continue
        if not _module_may_declare_models(module_dir):
            _warn(
                f'Модуль {module_dir.name} объявил {path}, '
                f'но не зависит от {OLLAMA_FRAMEWORK_MODULE} — пропуск'
            )
            continue
        collected.extend(
            _parse_models_file(
                path,
                module_dir_name=module_dir.name,
                env_values=env_values,
            )
        )

    return tuple(_dedupe_models(collected))


def aggregate_ollama_models(project_root: Path | str) -> OllamaModelsAggregate:
    root = Path(project_root).resolve()
    catalog = ModuleCatalog.from_env(root)
    agg = OllamaModelsAggregate()
    seen_modules: set[str] = set()

    for model in load_resolved_models(str(root)):
        agg.models.append(model)
        if model.source_module and model.source_module not in seen_modules:
            seen_modules.add(model.source_module)
            agg.modules.append(model.source_module)

    for module_dir in catalog.iter_module_dirs():
        path = module_dir / OLLAMA_MODELS_FILENAME
        if not path.is_file():
            continue
        if not _module_may_declare_models(module_dir):
            agg.skipped_modules.append(module_dir.name)

    return agg


def dump_ollama_models_json(project_root: Path | str) -> str:
    agg = aggregate_ollama_models(project_root)
    payload = {
        'models': [asdict(model) for model in agg.models],
        'modules': agg.modules,
        'skipped_modules': agg.skipped_modules,
    }
    return json.dumps(payload, ensure_ascii=False)


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    parser = argparse.ArgumentParser(
        description='Агрегат modules/*/ollama_models.yaml для setup-full',
    )
    parser.add_argument('--root', type=Path, default=None, help=t('help_root_path'))
    parser.add_argument('--json', action='store_true', help=t('help_json_stdout'))
    args = parser.parse_args(argv)

    root = (args.root or Path.cwd()).resolve()
    if args.json:
        print(dump_ollama_models_json(root))
        return 0

    agg = aggregate_ollama_models(root)
    if not agg.models:
        print(format_console('info', 'Файлы ollama_models.yaml не найдены или пусты'))
        return 0

    print(format_console('info', f'Найдено моделей для setup-full: {len(agg.models)}'))
    for model in agg.models:
        kind_suffix = f' ({model.kind})' if model.kind else ''
        required_suffix = '' if model.required else ' [optional]'
        print(f'  - {model.name}{kind_suffix}{required_suffix} ← {model.source_module}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
