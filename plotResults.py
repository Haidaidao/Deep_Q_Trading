from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import floor
from ensemble import ResultNewState
import global_config
import json

#Call it with the name of file plus the number of walks
# python plotResults.py results 2 
outputFile=str(sys.argv[1]) + ".pdf"

pdf=PdfPages(outputFile)

func_map = {

    "NewState": ResultNewState,

}

config = json.load(open('plotResultsConf.json', 'r'))

with open('numFile.txt', 'r', encoding='utf-8') as file:
    numFile = int(file.read()) 
numFiles=numFile+1
ensemble = func_map[config['type']]

plt.figure(figsize=(8, 12)) # Điều chỉnh kích thước cho 2 hàng

# Bảng Valid
plt.subplot(2, 1, 1) # Hàng 1, cột 1
plt.axis('off')
val, col = ensemble(numFiles,"valid",0)
t=plt.table(cellText=val, colLabels=col, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")

# Bảng Test
plt.subplot(2, 1, 2) # Hàng 2, cột 1
plt.axis('off')
val, col = ensemble(numFiles,"test",0)
t=plt.table(cellText=val, colLabels=col, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("RESULT", fontsize=10)
pdf.savefig()
pdf.close()
