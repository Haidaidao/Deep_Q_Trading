from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys
from math import floor
from ensemble import BaseRule, RandomForestEnsemble, EnsembleAuthor, XGBoostEnsemble
import global_config
import json

#Call it with the name of file plus the number of walks
# python plotResults.py results 2 
outputFile=str(sys.argv[1]) + ".pdf"

pdf=PdfPages(outputFile)

func_map = {
    "BaseRule": BaseRule,
    "Random": RandomForestEnsemble,
    "Author": EnsembleAuthor,
    "XGBoost": XGBoostEnsemble
}

numFiles=global_config.num_files + 1

ensemble = func_map[global_config.ensemble_type]

i=1
###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))

#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col = ensemble(numFiles,0,"valid",0)
t=plt.table(cellText=val, colLabels=col, fontsize=100, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")


plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0,"test",0)

t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("Full ensemble")
pdf.savefig()
###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))

#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col = ensemble(numFiles,0.9,"valid",0)
t=plt.table(cellText=val, colLabels=col, fontsize=100, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")


plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.9,"test",0)

t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("90%")
pdf.savefig()

###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))

#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col = ensemble(numFiles,0.8,"valid",0)
t=plt.table(cellText=val, colLabels=col, fontsize=100, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")


plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.8,"test",0)

t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("80%")
pdf.savefig()


###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))

#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col = ensemble(numFiles,0.7,"valid",0)
t=plt.table(cellText=val, colLabels=col, fontsize=100, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")


plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.7,"test",0)

t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("70%")
pdf.savefig()

###########-------------------------------------------------------------------|Tabella Ensemble|-------------------
x=2
y=1
plt.figure(figsize=(x*5,y*5))

#for i in range(1,floor(x*y/2)+1):
plt.subplot(y,x,i)
plt.axis('off')

val,col = ensemble(numFiles,0.6,"valid",0)
t=plt.table(cellText=val, colLabels=col, fontsize=100, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Valid")


plt.subplot(y,x,i+1)
plt.axis('off')

val,col=ensemble(numFiles,0.6,"test",0)

t=plt.table(cellText=val, colLabels=col, fontsize=30, loc='center')
t.auto_set_font_size(False)
t.set_fontsize(6)
t.auto_set_column_width(col=list(range(len(col))))
plt.title("Test")

plt.suptitle("60%")
pdf.savefig()

pdf.close()