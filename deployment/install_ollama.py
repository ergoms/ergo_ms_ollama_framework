"""
Установка Ollama в virtual_env/packages/ollama без Django.

Вызов: ergoms package-install ollama [--force] [--refresh]
       ergoms ollama_framework:install-ollama [--force] [--refresh]

Регистрация в реестре: modules/ollama_framework/packages.yaml
"""

from __future__ import annotations

import argparse
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEPLOYMENT_DIR = PROJECT_ROOT / 'core' / 'deployment'
if str(DEPLOYMENT_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOYMENT_DIR))

from console_tags import configure_stdio_utf8, format_console  # noqa: E402
from project_layout import (  # noqa: E402
    cache_downloads_dir,
    cache_tmp_dir,
    ensure_dir,
    packages_dir,
)

DOWNLOAD_CHUNK_SIZE = 8 * 1024 * 1024
PROGRESS_UPDATE_INTERVAL_SEC = 0.5
DOWNLOAD_USER_AGENT = 'ergoms/1.0 (Ollama installer)'
DOWNLOAD_TIMEOUT_SEC = 300
MIN_ARCHIVE_BYTES = 1024 * 1024


def ollama_dir(root: Path) -> Path:
    return packages_dir(root) / 'ollama'


def ollama_models_dir(root: Path) -> Path:
    return packages_dir(root) / 'ollama_models'


def ollama_exe_name() -> str:
    return 'ollama.exe' if platform.system().lower() == 'windows' else 'ollama'


def is_installed(root: Path) -> bool:
    base = ollama_dir(root)
    name = ollama_exe_name()
    return (base / name).is_file() or (base / 'bin' / name).is_file()


def _resolve_archive(machine: str) -> tuple[str, str]:
    system = platform.system().lower()
    machine = machine.lower()

    if system == 'windows':
        if 'arm64' in machine or 'aarch64' in machine:
            name = 'ollama-windows-arm64.zip'
        elif '64' in machine or 'x86_64' in machine or 'amd64' in machine:
            name = 'ollama-windows-amd64.zip'
        else:
            raise RuntimeError(f'Неподдерживаемая архитектура Windows: {machine}')
        return name, 'zip'

    if system == 'linux':
        # С релизов ~0.32 артефакты Linux — tar.zst (старый .tgz удалён → 404).
        if 'x86_64' in machine or 'amd64' in machine:
            name = 'ollama-linux-amd64.tar.zst'
        elif 'arm64' in machine or 'aarch64' in machine:
            name = 'ollama-linux-arm64.tar.zst'
        else:
            raise RuntimeError(f'Неподдерживаемая архитектура Linux: {machine}')
        return name, 'tar.zst'

    raise RuntimeError(f'Неподдерживаемая ОС: {system}')


def _archive_url(archive_name: str) -> str:
    return (
        'https://github.com/ollama/ollama/releases/latest/download/'
        f'{archive_name}'
    )


def _write_download_progress(
    downloaded: int,
    total_size: int,
    speed_bps: float,
) -> None:
    downloaded_mb = downloaded / (1024 * 1024)
    speed_mbps = speed_bps / (1024 * 1024)

    if total_size > 0:
        total_mb = total_size / (1024 * 1024)
        percent = downloaded / total_size * 100
        message = (
            f'Прогресс: {percent:.1f}% '
            f'({downloaded_mb:.1f} / {total_mb:.1f} МБ) '
            f'— {speed_mbps:.1f} МБ/с'
        )
    else:
        message = f'Скачано: {downloaded_mb:.1f} МБ — {speed_mbps:.1f} МБ/с'

    sys.stdout.write(f'\r{message}')
    sys.stdout.flush()


def _download_with_curl(url: str, destination: Path) -> bool:
    curl_exe = shutil.which('curl')
    if not curl_exe:
        return False

    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        result = subprocess.run(
            [
                curl_exe,
                '-L',
                '--fail',
                '--retry',
                '3',
                '--retry-delay',
                '2',
                '-A',
                DOWNLOAD_USER_AGENT,
                '-o',
                str(destination),
                url,
                '--progress-bar',
            ],
            check=False,
        )
    except OSError:
        return False

    return (
        result.returncode == 0
        and destination.is_file()
        and destination.stat().st_size >= MIN_ARCHIVE_BYTES
    )


def _download_with_urllib(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(
        url,
        headers={'User-Agent': DOWNLOAD_USER_AGENT},
    )
    downloaded = 0
    started_at = time.monotonic()
    last_ui_at = 0.0

    with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT_SEC) as response:
        total_size = int(response.headers.get('Content-Length') or 0)
        with destination.open('wb') as output_file:
            while True:
                chunk = response.read(DOWNLOAD_CHUNK_SIZE)
                if not chunk:
                    break
                output_file.write(chunk)
                downloaded += len(chunk)

                now = time.monotonic()
                is_complete = total_size > 0 and downloaded >= total_size
                if is_complete or now - last_ui_at >= PROGRESS_UPDATE_INTERVAL_SEC:
                    elapsed = max(now - started_at, 0.001)
                    _write_download_progress(downloaded, total_size, downloaded / elapsed)
                    last_ui_at = now

    print()
    if destination.stat().st_size < MIN_ARCHIVE_BYTES:
        destination.unlink(missing_ok=True)
        raise RuntimeError(f'Скачанный файл слишком маленький: {url}')


def _download_archive(url: str, destination: Path) -> None:
    print(f'Скачивание: {url}')
    if _download_with_curl(url, destination):
        print()
        return

    _download_with_urllib(url, destination)


def _locate_ollama_binary(target_dir: Path) -> Path | None:
    name = ollama_exe_name()
    direct = target_dir / name
    if direct.is_file():
        return direct
    for candidate in (
        target_dir / 'bin' / name,
        target_dir / 'lib' / 'ollama' / name,
    ):
        if candidate.is_file():
            return candidate
    matches = sorted(target_dir.rglob(name))
    return matches[0] if matches else None


def _extract_archive(archive_path: Path, target_dir: Path, archive_type: str) -> None:
    print('Распаковка...')
    target_dir.mkdir(parents=True, exist_ok=True)
    if archive_type == 'zip':
        with zipfile.ZipFile(archive_path, 'r') as archive:
            archive.extractall(target_dir)
    elif archive_type == 'tar.zst':
        zstd = shutil.which('zstd')
        if not zstd:
            raise RuntimeError(
                'Для распаковки Ollama нужен zstd '
                '(пакет zstd / ergoms package-install при наличии).',
            )
        result = subprocess.run(
            ['tar', '-I', zstd, '-xf', str(archive_path), '-C', str(target_dir)],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            detail = (result.stderr or result.stdout or '').strip()
            raise RuntimeError(
                f'Не удалось распаковать {archive_path.name}: {detail or result.returncode}',
            )
    else:
        with tarfile.open(archive_path, 'r:gz') as archive:
            archive.extractall(target_dir)

    exe = _locate_ollama_binary(target_dir)
    if exe is None:
        raise RuntimeError(
            f'После распаковки не найден {ollama_exe_name()} в {target_dir}. '
            'Проверьте архив релиза Ollama.'
        )
    if not os.access(exe, os.X_OK):
        exe.chmod(exe.stat().st_mode | 0o111)
    # Совместимость со старым layout (packages/ollama/ollama): symlink → bin/ollama
    dest = target_dir / ollama_exe_name()
    if exe.resolve() != dest.resolve():
        if dest.is_symlink() or dest.exists():
            dest.unlink()
        try:
            dest.symlink_to(exe.relative_to(target_dir))
        except OSError:
            dest.symlink_to(exe)


def _migrate_legacy_models_dir(root: Path) -> None:
    packages = packages_dir(root)
    legacy_models = packages / 'models'
    models_path = ollama_models_dir(root)
    if not legacy_models.is_dir():
        return

    ensure_dir(models_path)
    if any(models_path.iterdir()):
        print(format_console(
            'warning',
            f'Устаревший каталог {legacy_models} не перенесён — '
            f'в {models_path} уже есть данные',
        ))
        return

    for item in legacy_models.iterdir():
        shutil.move(str(item), str(models_path / item.name))

    shutil.rmtree(legacy_models, ignore_errors=True)
    print(format_console(
        'warning',
        f'Данные из packages/models перенесены в {models_path}',
    ))


def install_ollama(root: Path, *, force: bool = False, refresh: bool = False) -> int:
    configure_stdio_utf8()
    target = ollama_dir(root)
    models = ollama_models_dir(root)

    if is_installed(root) and not force:
        print(format_console(
            'warning',
            f'Ollama уже установлен в {target}\n'
            'Используйте --force для переустановки',
        ))
        return 0

    ensure_dir(packages_dir(root))
    ensure_dir(models)
    _migrate_legacy_models_dir(root)

    if force and target.exists():
        print('Удаление старой установки...')
        shutil.rmtree(target, ignore_errors=True)

    print('Установка Ollama в packages...')
    archive_name, archive_type = _resolve_archive(platform.machine())
    url = _archive_url(archive_name)
    cache_path = ensure_dir(cache_downloads_dir(root)) / archive_name
    tmp_base = ensure_dir(cache_tmp_dir(root))

    try:
        use_cache = (
            cache_path.is_file()
            and cache_path.stat().st_size >= MIN_ARCHIVE_BYTES
            and not refresh
        )
        if use_cache:
            print(format_console('info', f'Используется кэш: {cache_path}'))
            _extract_archive(cache_path, target, archive_type)
        else:
            with tempfile.TemporaryDirectory(dir=str(tmp_base)) as tmp_dir:
                archive_path = Path(tmp_dir) / archive_name
                _download_archive(url, archive_path)
                shutil.copy2(archive_path, cache_path)
                _extract_archive(archive_path, target, archive_type)
    except Exception as exc:
        print(format_console('error', f'Ошибка при установке Ollama: {exc}'), file=sys.stderr)
        return 1

    print(format_console('ok', f'Ollama распакован в {target}'))
    print(
        f'\nOllama установлен.\n\n'
        f'Бинарник: {target.absolute()}\n'
        f'Модели: {models.absolute()}\n\n'
        f'OLLAMA_MODELS задаётся автоматически при запуске через ergoms.\n'
        f'Запуск сервера: ergoms ollama_framework:start-ollama\n'
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Установка Ollama в virtual_env/packages')
    parser.add_argument(
        '--root',
        type=Path,
        default=PROJECT_ROOT,
        help='Корень проекта ERGO MS',
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Переустановить Ollama, даже если он уже установлен',
    )
    parser.add_argument(
        '--refresh',
        action='store_true',
        help='Скачать архив заново, игнорируя кэш в virtual_env/cache/downloads',
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    configure_stdio_utf8()
    args = build_parser().parse_args(argv)
    return install_ollama(args.root.resolve(), force=args.force, refresh=args.refresh)


if __name__ == '__main__':
    raise SystemExit(main())
