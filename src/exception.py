import sys


class ProjectException(Exception):
    """
    Custom exception class for the Employee Salary Prediction project.
    """

    def __init__(self, error_message, error_detail):

        _, _, exc_tb = error_detail.exc_info()

        if exc_tb is not None:

            file_name = exc_tb.tb_frame.f_code.co_filename
            line_number = exc_tb.tb_lineno

            message = (
                f"Error occurred in file: {file_name}\n"
                f"Line Number: {line_number}\n"
                f"Error Message: {error_message}"
            )

        else:

            message = str(error_message)

        super().__init__(message)


# -----------------------------
# Testing
# -----------------------------
