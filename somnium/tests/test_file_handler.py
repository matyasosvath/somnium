#!/usr/bin/env python

import unittest
from unittest.mock import patch, mock_open

from util.file_handler import FileHandler


class TestFileHandler(unittest.TestCase):
    

    def test_file_handler_open(self):
        path = ".\\disseration\\thesis.md"
        content = "Mock message to write to file"
        with patch('writer.file_handler.open',  mock_open()) as mocked_file:
            FileHandler().write(content)
        
            # assert if opened file on write mode 'a'
            mocked_file.assert_called_once_with(path, mode='a', encoding='utf-8')

            # assert if write(content) was called from the file opened
            # in another words, assert if the specific content was written in file
            mocked_file().write.assert_called_once_with(content)






if __name__ == '__main__':
    unittest.main()

