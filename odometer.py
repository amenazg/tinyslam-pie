import pandas as pd
import numpy as np

b=1 #b est la distance entre les deux roues, il faut l'ajuster. 
class odometer:
    angle = 0  
    position_x = 0
    position_y = 0
    # Parcourir les donn�es
    def odom_values(dl,dr):
        #dc est la distance parcourue par le point milieu entre les deux roues.
        dc=(dl+dr)/2
        # Estimer la variation d'angle
        thetha = (dr-dl)/b
        # Calculer la distance parcourue    
        # Estimer la nouvelle position
        position_x +=-dc*np.sin(thetha)
        position_y +=dc*np.cos(thetha)
    
        # Mettre à jour l'angle precedent
        angle += thetha
        return([position_x,position_y,angle])

    


