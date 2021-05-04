#!/usr/bin/env python3



class Writer:
    def __init__(self):
        pass

    def generic_writer(self, interpretation, mode='a'):
        with open('article.txt', mode=mode, encoding="utf-8") as article:
            article.write(interpretation)







# class Processor:
#     def __init__(self, reader, writer):
#         self.reader = reader
#         self.writer = writer

#     def process(self):
#         while True:
#             data = self.reader.readline()
#             if not data: 
#                 break
#         data = self.converter(data)
#         self.writer.write(data)

#     def converter(self, data):
#         assert False, 'converter must be defined' # Or raise exception


# class Uppercase(Processor):
#     def converter(self, data):
#         return data.upper()

if __name__ == '__main__':

    s = 'ez egy nmagyon jó hipotézis volt \n'

    w = Writer()
    w.generic_writer(s)

