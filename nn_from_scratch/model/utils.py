import os


def get_abs_path(relative_path):
    """
    Get the absolute path of the given relative path to the root directory of the project.

    Args:
        relative_path (str): The relative path to the root directory of the project.

    Returns:
        str: The absolute path of the given relative path to the root directory of the project.
    """
    project_root = os.environ.get('PROJECT_ROOT')
    if project_root is None:
        raise Exception("PROJECT_ROOT environment variable is not set")
    return os.path.abspath(os.path.join(project_root, relative_path)).replace('\\', '/')
