import os
import sys
import zipfile

def GetFilesList(dir:str):
    file_list = [dir]
    for root, dirs, files in os.walk(dir):
        for file in files:
            file_list.append(os.path.join(root, file))
        for d in dirs:
            file_list += GetFilesList(d)
    return file_list

if __name__ == '__main__':
    root = '.'
    report_path = ''
    for _, _, files in os.walk('.'):
        for file in files:
            if '.docx' in file:
                report_path = file.replace('.docx', '.pdf')
        break
    zipfile_name = report_path.replace('.pdf', '.zip')
    if os.path.exists(os.path.join(root, zipfile_name)):
        os.remove(os.path.join(root, zipfile_name))
    
    with zipfile.ZipFile(os.path.join(root, zipfile_name), 'w') as zip:
        zip.write(report_path)
        for path in sys.argv[1:]:
            if os.path.isfile(os.path.join(root, path)):
                zip.write(os.path.join(root, path))
            else:
                for p in GetFilesList(path):
                    zip.write(os.path.join(root, p))