import sys
from src.logger import logging

def error_message_detail(error,error_detail:sys):
    _,_,exc = error_detail.exc_info()
    file_name = exc.tb_frame.f_code.co_filename
    error_messages = "Error occurred in python script name [{0}] line number [{1}] error message [{2}]".format(
        file_name,exc.tb_lineno,str(error)
    )
    return error_messages

class CustomException(Exception):
    def __init__(self, error_messages,error_detail:sys):
        super().__init__(error_messages)
        self.error_message = error_message_detail(error_messages,error_detail=error_detail)
    
    def __str__(self):
        return self.error_message

