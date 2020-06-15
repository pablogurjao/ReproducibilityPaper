import paho.mqtt.client as mqtt
import sys
import tkinter as tk

#definicoes: 
hostname="127.0.0.1"
PortaBroker = 1883
KeepAliveBroker = 60
TopicoSubscribe = "ledStatus" 

global label, var

def _create_circle(self, x, y, r, **kwargs):
    return self.create_oval(x-r, y-r, x+r, y+r, **kwargs)

#Callback - conexao ao broker realizada
def on_connect(client, userdata, flags, rc):
    print("[STATUS] Conectado ao Broker. Resultado de conexao: "+str(rc))
 
    #faz subscribe automatico no topico
    client.subscribe(TopicoSubscribe)

#Callback - mensagem recebida do broker
def on_message(client, userdata, msg):
    global label, var
    RecievedMessage = int(msg.payload)
    

    if RecievedMessage is 0 :
        RED = "red"
        GREEN = "grey"
        var.set("\rDoor Closed")

    else:
        RED = "grey"
        GREEN = "green"
        var.set("\rDoor Oppened")

    label.pack()
    tk.Canvas.create_circle = _create_circle

    canvas.create_circle(50, 100, 30, fill=GREEN, outline="black", width=4)
    canvas.create_circle(150, 100, 30, fill=RED, outline="black", width=4)
    print(RecievedMessage)

#programa principal:
try:
    print("[STATUS] Inicializando MQTT...")
    #inicializa MQTT:
    client = mqtt.Client()
    client.on_connect = on_connect
    client.on_message = on_message

    client.connect(hostname, PortaBroker, KeepAliveBroker)
    

    root = tk.Tk()
    canvas = tk.Canvas(root, width=200, height=200, borderwidth=0, highlightthickness=0, bg="white")
    canvas.pack()

    var = tk.StringVar()
    label = tk.Label(root, textvariable=var, padx = 100)

    root.wm_title("Status Door")
    client.loop_start()
    root.mainloop()
    


except KeyboardInterrupt:
    client.loop_stop()
    print ("\nCtrl+C pressionado, encerrando aplicacao e saindo...")
    sys.exit(0)