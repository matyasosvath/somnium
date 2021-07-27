#!/usr/bin/env python3


class Luhmann:
    def __init__(self):
        pass

    def write(self, text, mode='a'):
        with open('disszertacio.txt', mode=mode, encoding="utf-8") as article:
            article.write(text)



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

    zettel = Luhmann()
    zettel.write("Tesztelése a generic wiriter functionnek")

