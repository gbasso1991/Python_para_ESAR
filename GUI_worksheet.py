#%% GUI_worksheet.py
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

root=tk.Tk()

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
Use the config() method with keyword attributes.

'''
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
        message='Boton de descarga clickeado!'
    )

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
