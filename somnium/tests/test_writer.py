#!/usr/bin/env python

import unittest
from unittest.mock import Mock
from unittest.mock import patch, mock_open


from writer.format_level import FormatLevel
from writer.Writer import Writer
from file_handler import FileHandler


class TestFileHandler(unittest.TestCase):
    
    def setUp(self):
        self.file_handler = FileHandler()
        self.writer = Writer(format_level=FormatLevel, file_handler=self.file_handler)

    def test_writer_heading(self):
        # Arrange
        expected = "# Mock Heading"
        # Act
        actual = self.writer.format_text(FormatLevel.HEADING, "Mock Heading")
        # Assert
        self.assertEqual(actual, expected)

    def test_writer_text(self):
        # Arrange
        expected = "Mock message goes here"
        # Act
        actual = self.writer.format_text(FormatLevel.TEXT, "Mock message goes here")
        # Assert
        self.assertEqual(actual, expected)

    def test_writer_math(self):
        # Arrange
        expected = f"$$a^2+b^2=c^2$$"
        # Act
        actual = self.writer.format_text(FormatLevel.MATH, "a^2+b^2=c^2")
        # Assert
        self.assertEqual(actual, expected)

    def test_with_level(self):
        # Arrange
        expected = True
        # Act
        actual_heading = self.writer.with_level(FormatLevel.HEADING)
        actual_text = self.writer.with_level(FormatLevel.TEXT)
        actual_math = self.writer.with_level(FormatLevel.MATH)
        # Assert
        self.assertEqual(actual_heading, expected)
        self.assertEqual(actual_text, expected)
        self.assertEqual(actual_math, expected)

    def test_write(self):
        path = ".\\disseration\\thesis.md"
        content = "Heading 1"
        with patch('writer.Writer.open',  mock_open()) as mocked_file:
            Writer(FormatLevel, self.file_handler).write(FormatLevel.TEXT, content)
        
            # assert if opened file on write mode 'a'
            mocked_file.assert_called_once_with(path, mode='a', encoding='utf-8')

            # assert if write(content) was called from the file opened
            # in another words, assert if the specific content was written in file
            mocked_file().write.assert_called_once_with(content)



if __name__ == '__main__':
    unittest.main()

