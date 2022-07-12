#!/usr/bin/env python

import os
import matplotlib.pyplot as plt

from ..logging.ilogger import ILogger


class FileHandler(object):

    PROJECT_ROOT_DIR = "."
    THESIS_PATH = os.path.join(PROJECT_ROOT_DIR, "disseration")
    THESIS_NAME = "thesis.md"
    IMAGES_PATH = os.path.join(PROJECT_ROOT_DIR, "figures")
    
    def __init__(self, logger: ILogger) -> None:

        self.logger = logger

        os.makedirs(self.THESIS_PATH, exist_ok=True)
        os.makedirs(self.IMAGES_PATH, exist_ok=True)

    def write(self, text, mode='a'):
        with open(
            self.THESIS_PATH + self.THESIS_NAME, 
            mode=mode, 
            encoding="utf-8") as thesis:
            
            thesis.write(text + "\n")
    
    def save_figure(self, fig_name, tight_layout=True, 
        fig_extension="png", resolution=300) -> None:
        """
        """
        plt.figure(figsize=(14, 8)) # Figure size
        
        path = os.path.join( # Create path
            self.IMAGES_PATH, fig_name + "." + fig_extension
            )
        # logger.log()
        if tight_layout:
            plt.tight_layout()
        
        # Save figure
        plt.savefig(path, format=fig_extension, dpi=resolution)
        
        return None 



