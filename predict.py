from tkinter.filedialog import *

diary = ""
root = Tk()
root.title('Fill the image for you')

filepath = askopenfilename(title="Ouvrir une image", filetypes=[('all files', '.*')])
photo = PhotoImage(file=filepath)
photo.width()
canvas = Canvas(root, height=photo.height() + 20, width=photo.width() + 20)
canvas.pack(side=LEFT, fill=BOTH, expand=1)
canvas.create_image(100, 100, anchor=NW, image=photo)

caption1 = Label(root, text="Enter a caption")
ent = Entry(root)

caption1.pack()
ent.pack()
# lbl2.pack()
# ent2.pack()
root.mainloop()

"""
from tkinter import *

fenetre = Tk()
photo = PhotoImage(file=filepath)
canvas = Canvas(fenetre, width=photo.width(), height=photo.height(), bg="yellow")
canvas.create_image(0, 0, anchor=NW, image=photo)
canvas.pack()
"""
