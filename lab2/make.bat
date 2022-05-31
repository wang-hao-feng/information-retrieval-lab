@echo off
    ::生成报告
    python ../../docx2pdf/docx2pdf.py
    ::压缩文件
    python ../package.py preprocessed.py question_classification.py answer_sentence_selection.py answer_span_selection.py test_answer.json