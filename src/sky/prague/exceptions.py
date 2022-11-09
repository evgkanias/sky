class DatasetNotFoundWarning(Exception):
    """
    Exception thrown by the initialize method if the passed dataset file could not be found.
    """
    def __init__(self, filename: str):
        super().__init__(f"Dataset file '{filename}' not found.")


class DatasetReadException(Exception):
    """
    Exception thrown by the initialize method if an error occurred while reading the passed dataset file
    """
    def __init__(self, parameter_name: str):
        super().__init__(f"Dataset reading failed at '{parameter_name}'.")


class NoPolarisationWarning(Warning):
    """
    Exception thrown by the polarisation method if the dataset passed to the initialize method does not
    contain polarisation data.
    """
    def __init__(self):
        super().__init__("The supplied dataset does not contain polarisation data.")


class NotInitialisedException(Exception):
    """
    Exception thrown when using the model without calling the initialize method first.
    """
    def __init__(self):
        super().__init__("The model is not initialized.")
