@echo off
    ::生成报告
    python ../../docx2pdf/docx2pdf.py
    ::压缩文件
    python ../package.py retrival_system.py Ui_MainUI.py