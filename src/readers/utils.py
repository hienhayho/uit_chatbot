import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.constants import SUPPORTED_FILE_EXTENSIONS, SUPPORTED_LINK_EXTENSIONS
from src.utils import get_formatted_logger

logger = get_formatted_logger(__file__)


def check_valid_extenstion(file_path: str | Path) -> bool:
    """
    Check if the file extension is supported

    Args:
        file_path (str | Path): File path to check

    Returns:
        bool: True if the file extension is supported, False otherwise.
    """
    return Path(file_path).suffix in SUPPORTED_FILE_EXTENSIONS


def check_valid_link(link: str) -> bool:
    """
    Check if the link is supported

    Args:
        link (str): Link to check

    Returns:
        bool: True if the link is supported, False otherwise.
    """
    return link.split(":")[0] in SUPPORTED_LINK_EXTENSIONS


def get_files_from_folder_or_file_paths(files_or_folders: list[str]) -> list[str]:
    """
    Get all files from the list of file paths or folders

    Args:
        files_or_folders (list[str]): List of file paths or folders

    Returns:
        list[str]: List of valid file paths.
    """
    files = []

    for file_or_folder in files_or_folders:
        if Path(file_or_folder).is_dir():
            files.extend(
                [
                    str(file_path.resolve())
                    for file_path in Path(file_or_folder).rglob("*")
                    if check_valid_extenstion(file_path)
                ]
            )

        else:
            if check_valid_extenstion(file_or_folder):
                files.append(str(Path(file_or_folder).resolve()))
            else:
                logger.warning(f"Unsupported file extension: {file_or_folder}")

    return files


def get_files_from_list_links(links: list[str]) -> list[str]:
    """
    Get all links from the list of links

    Args:
        links (list[str]): List of links

    Returns:
        list[str]: List of valid file paths.
    """
    links = []

    for link in links:
        print(link)
        print("=========================")
        if check_valid_link(link):
            links.append(link)
        else:
            logger.warning(f"Unsupported link extension: {link}")

    return links
