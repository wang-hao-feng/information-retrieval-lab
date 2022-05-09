@echo off
    ::生成报告
    python ../../docx2pdf/docx2pdf.py
    ::压缩文件
    python ../package.py craw.py segment.py preprocess.json