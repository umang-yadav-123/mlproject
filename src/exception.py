import sys
from src.loggers import logging


def error_message_details(error, error_detail: sys):
    _, _, exc_tb = error_detail.exc_info()

    return (
        f"Error occurred in script name "
        f"[{exc_tb.tb_frame.f_code.co_filename}] "
        f"line number [{exc_tb.tb_lineno}] "
        f"error message [{str(error)}]"
    )


class customException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = error_message_details(error_message, error_detail)

    def __str__(self):
        return self.error_message