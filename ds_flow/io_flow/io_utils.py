



def list_files_all_subdirectories(dirName) -> list:
    """
    Takes a directory and returns a list of all files in all subdirectories.
    """
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + list_files_all_subdirectories(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles


def list_all_subdirectories(dirName) -> list:
    """
    Takes a directory and returns a list of all subdirectories.

    """
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    folders = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        if os.path.isdir(fullPath):
            folders.append(fullPath)
            folders += list_all_subdirectories(fullPath)
                
    return folders