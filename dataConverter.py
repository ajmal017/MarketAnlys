import sys
import xlrd  
import xlsxwriter

oldBook = xlrd.open_workbook('data/' + sys.argv[1]) 
oldSheet = oldBook.sheet_by_index(0)
newBook = xlsxwriter.Workbook('convertedData/' + sys.argv[1])
newSheet = newBook.add_worksheet()
newRowCount = 0 
for row in range(oldSheet.nrows):    
    a = oldSheet.row(row)
    py_date = xlrd.xldate.xldate_as_datetime(a[0].value, oldBook.datemode)
    if (py_date.minute % 5 == 0):
        for i in range(1, oldSheet.ncols - 1):
            value = oldSheet.cell_value(row,i)
            if (sys.argv[2] == 'invert'):
                value = 1.0/value
            newSheet.write(newRowCount, i, value)
        newSheet.write(newRowCount, 0, py_date.isoformat('-'))
        newRowCount += 1

newBook.close()