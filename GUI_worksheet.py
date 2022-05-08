#%% GUI_worksheet.py
# siguiendo tutorial de https://www.pythontutorial.net/tkinter/tkinter-pack/
import tkinter as tk
from turtle import window_height
from matplotlib.pyplot import text
#%%
root = tk.Tk() #creo la instancia
#In Tkinter, components are called widgets
#create a Label widget placed on the root window
msj = tk.Label(root,text='Una GUI')
msj.pack(expand=False)# positions the Label on the main window:
#% #para arreglar borrosidad en windows
root.title('El titulo de la ventana')
root.geometry('600x400+100+50') #600x400 px and position of the window to 100/50 px from top/left of the screen

try:
    from ctypes import windll
    windll.shcore.SetProcessDpiAwareness(1)
finally:
    root.mainloop()

#%% Centrada en pagina 
root = tk.Tk() #creo la instancia
root.title('Cuadro centrado en pantalla')
window_width=500
window_height=200

#get dimensiones de la pantalla
screen_width=root.winfo_screenwidth()
screen_height=root.winfo_screenheight()

#posicion del centro
center_x= int(screen_width/2 - window_width/2)
center_y = int(screen_height/2 - window_height/2)

#set position
root.geometry(f'{window_width}x{window_height}+{center_x}+{center_y}')
#root.resizable(False, False) #para evitar reescaleo
#root.attributes('-alpha', 0.8) #opacidad
root.attributes('-topmost', 1) # para que este on top
root.mainloop() #keeps the window visible on the screen until you close it.

'''Use the title() method to change the title of the window.
Use the geometry() method to change the size and location of the window.
Use the resizable() method to specify whether a window can be resizable horizontally or vertically.
Use the window.attributes('-alpha',0.5) to set the transparency for the window.
Use the window.attributes('-topmost', 1) to make the window always on top.
Use lift() and lower() methods to move the window up and down of the window stacking order.
Use the iconbitmap() method to change the default icon of the window.'''
#%% Tk themed widgets by using the Tkinter.ttk module (tk themed).
import tkinter as tk
from tkinter import ttk

tk.Label(root,text='Label clasico').pack()
ttk.Label(root,text='Label themed').pack()
root.attributes('-topmost', 1) # para que este on top
root.mainloop()

'''Tkinter has both classic and themed widgets. The Tk themed widgets are also known as ttk widgets.
The tkinter.ttk module contains all the ttk widgets.
Do use ttk widgets whenever they’re available.

Ttk widgets provide you with three ways to set options:

Use keyword arguments at widget creation.
Use a dictionary index after widget creation.
Use the config() method with keyword attributes.'''
#%% comandos interactivos
import tkinter as tk
from tkinter import ttk

def button_clicked():
    print('Button clicked')

def select(opcion):
    print(opcion)

root=tk.Tk()
root.title('Ventana interactiva')
root.geometry('600x400+100+50') #600x400 px and position of the window to 100/50 px from top/left of the screen
root.attributes('-topmost', 1) # para que este on top

#boton=ttk.Button(root,text='Clickeame',command=button_clicked,)
#boton.pack(expand=True)

ttk.Button(root,text='Piedra',command=lambda:select('Piedra')).pack()
ttk.Button(root,text='Papel',command=lambda:select('Papel')).pack()
ttk.Button(root,text='Tijera',command=lambda:select('Tijera')).pack()
root.mainloop()

'''Assign a function name to the command option of a widget is called command binding in Tkinter.
The assigned function will be invoked automatically when the corresponding event occurs on the widget.
Only few widgets support the command option.

Use the bind() method to bind an event to a widget.
Tkinter supports both instance-level and class-level bindings.
'''
#%% Label widget is used to display a text or image on the screen
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry('300x200')
root.resizable(False, False)
root.title('Label Widget Demo')
# show the label here
label = ttk.Label(root, text='Esta es una "label"',font=("Helvetica", 14))

label.pack(ipadx=10, ipady=10)

root.mainloop()
#%%
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.geometry('600x450')
root.resizable(False, False)
root.title('Label Widget Demo')

# display an image label
foto = tk.PhotoImage(file='./python_logo.png')

image_label = ttk.Label(
    root,
    image=foto,
    text='Python logo',
    compound='top')
image_label.pack()#ipadx=10, ipady=10)

root.mainloop()

#%% Boton
import tkinter as tk
from tkinter import ttk

# root window
root = tk.Tk()
root.geometry('300x200')
root.resizable(False, False)
root.title('Button Demo')

#boton de exit()
exit_button = ttk.Button(root,
text='Cerrar',
command=lambda: root.quit())
exit_button.pack()
root.mainloop()
'''Este no funca bien, se cuelga el programa'''
#%% Boton para descargar

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
# root window
root = tk.Tk()
root.geometry('300x200')
root.resizable(False, False)
root.title('Image Button Demo')

#boton de descarga
def download_clicked():
    showinfo(
        title='Info',   
        message='Boton de descarga clickeado!'    )

download_icon = tk.PhotoImage(file='./download.png')

download_button = ttk.Button(root,
    image=download_icon,
    command=download_clicked)

download_button.pack(
    ipadx=50,
    ipady=5,
    expand=True)

root.mainloop()    
# %%To display both text and image on a button, you need to use the compound option.
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

root = tk.Tk()
root.geometry('300x200')
root.resizable(False, False)
root.title('Image Button Demo')

#handler del boton
def download_clicked():
    showinfo(
        title='Info',   
        message='Boton de descarga clickeado!')

download_icon = tk.PhotoImage(file='./download.png')

download_button = ttk.Button(
    root,
    image=download_icon,
    text='Download',
    compound=tk.LEFT,
    command=download_clicked)

download_button.pack(
    ipadx=5,
    ipady=5,
    expand=True)

root.mainloop() 
'''Use the ttk.Button() class to create a button.
Assign a lambda expression or a function to the command option to respond to the button click event.
Assign the tk.PhotoImage() to the image property to display an image on the button.
Use the compound option if you want to display both text and image on a button.'''
#%% Entry widget
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

# root window
root = tk.Tk()
root.geometry("300x150")
root.resizable(False, False)
root.title('Logueate wachin')

# guardar email address y password
email = tk.StringVar()
password = tk.StringVar()

def login_clicked():
    '''Callback cuando el boton es cliqueado'''
    msj= f'Se ha ingresado:\n  mail: {email.get()}\n  pass: {password.get()}'
    showinfo(title='Informacion',
    message=msj)

# Frame de logueo
signin= ttk.Frame(root)
signin.pack(padx=10,pady=10,fill='x',expand=True)

#mail
email_label=ttk.Label(signin,text='E-mail:')
email_label.pack(fill='x',expand=True)

email_entry = ttk.Entry(signin,textvariable=email)
email_entry.pack(fill='x',expand=True)
email_entry.focus()

#password
password_label=ttk.Label(signin,text='Password:')
password_label.pack(fill='x',expand=True) 

password_entry= ttk.Entry(signin,textvariable=password,show='*')
password_entry.pack(fill='x',expand=True)

#login Button
login_button = tk.Button(signin,text='Login',command=login_clicked)
login_button.pack(fill='x',expand=True,pady=10)

root.mainloop()

'''Use the ttk.Entry widget to create a textbox.
Use an instance of the StringVar() class to associate the current text of the Entry widget with a string variable.
Use the show option to create a password entry.'''

#%% Geometria
''' To arrange widgets on a window, you use geometry managers. 
The pack() method is one of three geometry managers in Tkinter.
The other geometry managers are grid() and place()
The pack geometry manager has many configurations. 
The following are the most commonly used options: 
fill, expand, side, ipadx, ipady, padx, and pady'''

import tkinter as tk

root = tk.Tk()
root.title('Pack Demo')
root.geometry("300x200")

box1= tk.Label(root, text='Box 1',bg='green',fg='black')
box1.pack(ipadx=10,ipady=10,fill='both',expand=True)

box2= tk.Label(root, text='Box 2',bg='red',fg='white')
box2.pack(ipadx=10,ipady=10,fill='both',expand=True)


root.mainloop()
#%% The side option specifies the alignment of the widget. 
# It can be 'left', 'top', 'right', and 'bottom'.
import tkinter as tk

root = tk.Tk()
root.title('Pack Demo')
root.geometry("300x200")

box1= tk.Label(root, text='Box 1',bg='green',fg='black')
box1.pack(ipadx=10,ipady=10,fill='both',expand=True,side='left')

box2= tk.Label(root, text='Box 2',bg='red',fg='white')
box2.pack(ipadx=10,ipady=10,fill='both',expand=True,side='right')
root.mainloop()

'''The geometry manager is suitable for the following:
Placing widgets in a top-down layout.
Placing widgets side by side'''
#%%
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.title('Pack Demo')
root.geometry("300x200")

# alinear widgets top-down
label1 = tk.Label(root,text='Box 1',bg="red",fg="white")
label1.pack(ipadx=10,ipady=10,fill='x')
label2 = tk.Label(root,text='Box 2',bg="green",fg="white")
label2.pack(ipadx=10,ipady=10,fill='x')
label3 = tk.Label(root,text='Box 3',bg="blue",fg="white")
label3.pack(ipadx=10,ipady=10,fill='x')

#alinear widgets left-right.
label4=tk.Label(root,text='Left',bg="cyan",fg="black")
label4.pack(expand=True,fill='both',side='left')

label5=tk.Label(root,text='Center',bg="magenta",fg="black")
label5.pack(expand=True,fill='both',side='left')

label6=tk.Label(root,text='Right',bg="yellow",fg="black")
label6.pack(expand=True,fill='both',side='left')
root.mainloop()
'''Use Tkinter pack geometry manager to arrange widgets in a 
top-down layout or side by side.
Use the fill, expand, and side options of pack geometry 
manager to control how widgets arranged.'''

#%% Grids
'''Each row and column in the grid is identified by an index. 
By default, the first row has an index of zero, the second row 
has an index of one, and so on. Likewise, the columns in the 
grid have indexes of zero, one, two, etc.
The indexes of rows and columns in a grid don’t have to start
 at zero. In addition, the row and column indexes can have gaps.'''

import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo
root = tk.Tk()
root.title('Grid Demo')
root.geometry("240x100")
root.title('Login')
root.resizable(0,0)

def login_clicked():
    '''Callback cuando el boton es cliqueado'''
    msj= f'Se ha ingresado:\n  mail: {email.get()}\n  pass: {password.get()}'
    showinfo(title='Informacion',    message=msj)

#Grid config
root.columnconfigure(0,weight=1)
root.columnconfigure(1,weight=3)


#para recuperar params
email = tk.StringVar()
password = tk.StringVar()

#User
username_label = ttk.Label(root,text='Username:')
username_label.grid(column=0,row=0,sticky=tk.W,padx=5,pady=5)

username_entry = tk.Entry(root,textvariable=email)
username_entry.grid(column=1,row=0,sticky=tk.E, padx=5, pady=5)

#Pass
pass_label=tk.Label(root,text='Pass:')
pass_label.grid(column=0,row=1,sticky=tk.W,padx=5,pady=5)

pass_entry = tk.Entry(root,show='*',textvariable=password)
pass_entry.grid(column=1,row=1,sticky=tk.E, padx=5, pady=5)


#Boton de Login
login_button = ttk.Button(root,text="Confirm",command=login_clicked)
login_button.grid(column=1,row=3,sticky=tk.E,padx=5,pady=5)

root.mainloop()

#%%Ejemplo pero orientado a objetos
import tkinter as tk
from tkinter import ttk
from tkinter.messagebox import showinfo

class App(tk.Tk):
    def __init__(self):
        super().__init__()

        self.geometry('240x100')
        self.title('Login')
        self.resizable(0,0)

        #Grid
        self.columnconfigure(0,weight=1)
        self.columnconfigure(1,weight=3)

        self.user = tk.StringVar()
        self.password = tk.StringVar()

        self.create_widgets()
    
    def login_clicked(self):
        '''Callback cuando el boton es cliqueado'''
        msj= f'Se ha ingresado:\n  mail: {self.user.get()}\n  pass: {self.password.get()}'
        showinfo(title='Informacion',    message=msj)


    def create_widgets(self):
        #username
        username_label = ttk.Label(self,text='Username:')
        username_label.grid(column=0,row=0,sticky=tk.W,padx=5,pady=5)

        username_entry = tk.Entry(self,textvariable=self.user)
        username_entry.grid(column=1,row=0,sticky=tk.E, padx=5, pady=5)

        #Pass
        pass_label=tk.Label(self,text='Pass:')
        pass_label.grid(column=0,row=1,sticky=tk.W,padx=5,pady=5)

        pass_entry = tk.Entry(self,show='*',textvariable=self.password)
        pass_entry.grid(column=1,row=1,sticky=tk.E, padx=5, pady=5)

        #Boton de Login
        login_button = ttk.Button(self,text="Confirm",command=self.login_clicked)
        login_button.grid(column=1,row=3,sticky=tk.E,padx=5,pady=5)


if __name__=='__main__':
    app=App()
    app.mainloop()

'''Use the columnconfigure() and rowconfigure() methods to specify the weight of a column and a row of a grid.
Use grid() method to position a widget on a grid.
Use sticky option to align the position of the widget on a cell and define how the widget will be stretched.
Use ipadx, ipady and padx, pady to add internal and external paddings.'''    
#%% Text
from tkinter import Tk, Text
root= Tk()
root.resizable(0,0)
root.title('Text Widget Example')

text=Text(root,height=15)
text.insert('4.1','Texto de prueba')
text_content = text.get('1.0','end')
text['state'] = 'normal'#para desactivarlo usar disabled 
text.pack()

root.mainloop()

#%% Scrollbar 
import tkinter as tk
from tkinter import ttk

root = tk.Tk()
root.resizable(False, False)
root.title("Scrollbar Widget Example")

# apply the grid layout
root.grid_columnconfigure(0, weight=1)
root.grid_rowconfigure(0, weight=1)

inicial = '''Create a scrollbar with ttk.Scrollbar(orient, command)
The orient can be 'vertical' or 'horizontal'
The command can be yview or xview property of the scrollable widget that links 
to the scrollbar.
Set the yscrollcommand property of the scrollable widget so it links to the 
scrollbar.'''


# create the text widget
text = tk.Text(root, height=10)
text.grid(row=0, column=0, sticky='ew')
text.insert('1.0',inicial)
#creo scrollbar
scrollbar= ttk.Scrollbar(root,orient='vertical',command=text.yview)
scrollbar.grid(row=0,column=1,sticky='ns')

#comunicacion con la scrollbar
text['yscrollcommand']=scrollbar.set

root.mainloop()
#%%
# Tkinter ScrolledText widget 
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

root = tk.Tk()
root.title("ScrolledText Widget")
text_init = '''To make it more convenient, Tkinter provides you with the 
ScrolledText widget which does the same things as a text widget linked 
to a vertical scroll bar.

To use the ScrolledText widget, you need to import the ScrolledText class 
from the tkinter.scrolledtext module.

Technically, the ScrolledText class inherits from the Text class.

The ScrolledText widget uses a Frame widget inserted between the container 
and the Text widget to hold the Scrollbar widget.

Therefore, the ScrolledText has the same properties and methods as the 
Text widget. In addition, the geometry manager methods including pack, grid, 
and place are restricted to the Frame.'''

st= ScrolledText(root,width=80,height=10)
st.insert('1.0',text_init)
st.pack(fill=tk.BOTH,side=tk.LEFT,expand=True)

root.mainloop()

#%%
import tkinter as tk
from tkinter.scrolledtext import ScrolledText

text_init = '''To make it more convenient, Tkinter provides you with the 
ScrolledText widget which does the same things as a text widget linked 
to a vertical scroll bar.

To use the ScrolledText widget, you need to import the ScrolledText class 
from the tkinter.scrolledtext module.

Technically, the ScrolledText class inherits from the Text class.

The ScrolledText widget uses a Frame widget inserted between the container 
and the Text widget to hold the Scrollbar widget.

Therefore, the ScrolledText has the same properties and methods as the 
Text widget. In addition, the geometry manager methods including pack, grid, 
and place are restricted to the Frame.'''

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("ScrolledText Widget")
        st = ScrolledText(self, width=80, height=10)
        st.insert('1.0',text_init)
        st.pack(fill=tk.BOTH, side=tk.LEFT, expand=True)


if __name__ == "__main__":
    app = App()
    app.mainloop()

#%% Separador   
import tkinter as tk
from tkinter import ttk

info = '''Use a separator widget to place a thin horizontal or vertical 
rule between groups of widgets.
Remember to set the fill or sticky property to adjust the size of the separator.'''
root = tk.Tk()
root.geometry('400x200')
root.resizable(False, False)
root.title('Separator Widget Demo')

ttk.Label(root, text="Label 1").pack()

separador = ttk.Separator(root,orient='horizontal')
separador.pack(fill='x')
ttk.Label(root,text=info).pack()

root.mainloop()