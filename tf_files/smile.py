import sys
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import messagebox
from PIL import ImageTk, Image

import os, sys
import tensorflow as tf
#-----------------------------------------------------

#make frame
root = Tk()
root.geometry("400x175")

window = Toplevel()

#-----------------------------------------------------

def loadFile():
	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

	# change this as you see fit
	#image_path = sys.argv[1]
	global filename
	filename = askopenfilename()
	showImage()
   
def showImage():
	my_image = Image.open(filename)
	hpercent = (300 / float(my_image.size[1]))
	wsize = int((float(my_image.size[0]) * float(hpercent)))
	my_image = my_image.resize((wsize, 300), Image.ANTIALIAS)
	my_image = ImageTk.PhotoImage(my_image)
	canvas.create_image(250, 150, anchor=CENTER, image=my_image)
	canvas.my_image = my_image


def classifySmile():
	# Loads label file, strips off carriage return
	label_lines = [line.rstrip() for line 
                   in tf.gfile.GFile("retrained_labels.txt")]
				   
	#image_data = tf.gfile.FastGFile(image_path, 'rb').read()
	image_data = tf.gfile.FastGFile(filename, 'rb').read()

	# Unpersists graph from file
	with tf.gfile.FastGFile("retrained_graph.pb", 'rb') as f:
		graph_def = tf.GraphDef()
		graph_def.ParseFromString(f.read())
		tf.import_graph_def(graph_def, name='')

	with tf.Session() as sess:
		# Feed the image_data as input to the graph and get first prediction
		softmax_tensor = sess.graph.get_tensor_by_name('final_result:0')
		
		predictions = sess.run(softmax_tensor, \
				 {'DecodeJpeg/contents:0': image_data})
		
		# Sort to show labels of first prediction in order of confidence
		top_k = predictions[0].argsort()[-len(predictions[0]):][::-1]
		
		text = Text(window)
		for node_id in top_k:
			human_string = label_lines[node_id]
			score = predictions[0][node_id]
			smile = '%s (score = %.5f)' % (human_string, score)
			#messagebox.showinfo( "Smile Classification", smile)
			
			
			
			text.insert(INSERT, smile + "\n")
			text.pack()
			#text.insert(END, "Bye Bye.....")
		

	
top_frame = Frame(root)
bottom_frame = Frame(root)

top_frame.pack()
bottom_frame.pack()

#headline
headline = Label(top_frame, text="Genuine Smile App", bg='white', fg='green')
headline.config(font=('Arial', 27))
headline.grid(padx=10, pady=10)




#loadfile_button
load_button = Button(bottom_frame, text="Load Image", bg='white', fg='green', command=loadFile)
load_button.config(height = 1, width = 20)
load_button.grid(sticky = S)

#enter_button
detect_button = Button(bottom_frame, text="Check your smile", bg='white', fg='green', command=classifySmile)
detect_button.config(height = 1, width = 20)
detect_button.grid(sticky = S)

#quit_button
quit_button = Button(bottom_frame, text="Exit", bg="white", fg="green", command=quit)
quit_button.config(height = 1, width = 20)
quit_button.grid(sticky = S)

canvas = Canvas(window, width = 500, height = 400)
canvas.pack()

fcanvas = Canvas(window, width = 50, height = 50)
fcanvas.pack()

root.mainloop()
