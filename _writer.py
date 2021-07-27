#!/usr/bin/env python3



class Writer:
    def __init__(self):
        pass

    def generic_writer(self, interpretation, mode='a'):
        with open('disszertacio.txt', mode=mode, encoding="utf-8") as article:
            article.write(interpretation)

# Define a szakdolgozat.md
def iras(sima_text):
    with open('szakdolgozat.md', 'a') as f:
        f.write(sima_text)
        f.write('\n \n')



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


    w = Writer()
    w.generic_writer("Tesztelése a generic wiriter functionnek")

