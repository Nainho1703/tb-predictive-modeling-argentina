#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 14:20:28 2023

@author: leonardo
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# import shapefile as shp  # Requires the pyshp package
# import matplotlib.pyplot as plt
import geopandas as gpd
from fuzzywuzzy import process
import numpy as np
# ==========================================================================
# ============== Replaces %90 and more similar strings ===================== 
# ==========================================================================
def replace_similar_values(value):
    # print('Reescribiendo departamentos')
    for standard_value in list(localidades):
        # print(standard_value)
        if process.extractOne(value, [standard_value])[1] > 80:
            return standard_value
    return value
# ==========================================================================
# ==========================================================================

# sf = shp.Reader("/home/leonardo/MEGAsync/Datos Argentina/Codgeo_Pais_x_dpto_con_datos/pxdptodatosok.shp")
# 
# print("Initializing Shapefile")
# # sf = shp.Reader("ap_abl")
# apShapes = sf.shapes()
# points = apShapes[3].points
# print("Shapefile Initialized")

# print("Initializing Display")
# fig = plt.figure()
# ax = fig.add_subplot(111)
# # plt.xlim([78, 79])
# # plt.ylim([19, 20])
# print("Display Initialized")

# print("Creating Polygon")
# ap = plt.Polygon(points, fill=False, edgecolor="k")
# ax.add_patch(ap)
# print("Polygon Created")

# print("Displaying polygon")
# plt.show()

# plt.figure()
# for shape in sf.shapeRecords():
#     x = [i[0] for i in shape.shape.points[:]]
#     y = [i[1] for i in shape.shape.points[:]]
#     plt.plot(x,y)
# plt.show()




DF1 = pd.read_excel('base2009_2018_Leonardo_Lopez.xlsx')
DF2 = pd.read_excel('base_2019_2022_Leonardo_Lopez_al_01_10_2023.xlsx')
DF=pd.concat([DF1,DF2])

shape=gpd.read_file('/home/leonardo/MEGAsync/Datos Argentina/Codgeo_Pais_x_dpto_con_datos/pxdptodatosok.shp')

# Limpiar Bse de Datos

ProvinciasMapa=shape['provincia'].unique()
DF['PROVINCIA_RESIDENCIA']=DF['PROVINCIA_RESIDENCIA'].replace('CABA', 'Ciudad Autónoma de Buenos Aires')
DF=DF.dropna()
for i in ProvinciasMapa:
    localidades=shape[shape['provincia']==i]['departamen'].unique()
    DF[DF['PROVINCIA_RESIDENCIA']==i]['DEPARTAMENTO_RESIDENCIA'] = DF[DF['PROVINCIA_RESIDENCIA']==i]['DEPARTAMENTO_RESIDENCIA'].apply(replace_similar_values)

localidades=shape['departamen'].unique()
DF['DEPARTAMENTO_RESIDENCIA'] = DF['DEPARTAMENTO_RESIDENCIA'].apply(replace_similar_values)



Provincias=DF['PROVINCIA_RESIDENCIA'].unique()

Provincias=np.delete(Provincias,22)
# Provincias.remove('Tierra del Fuego')

shape=shape[shape['provincia'].isin(Provincias)]
shape["hogares"] = pd.to_numeric(shape['hogares'], downcast='integer')
shape["personas"] = pd.to_numeric(shape['personas'], downcast='integer')

DF.to_csv('DatosFiltrados.csv',sep=',')

Anios=DF['YEAR'].unique()
Sex=DF['SEXO'].unique()
DF['FECHA_NOTIFICACION']=pd.to_datetime(DF['FECHA_NOTIFICACION'])
# DF.boxplot(column="EDAD_DIAGNOSTICO",by="SEXO")

DF2018=DF[DF['YEAR']==2018]
DF2018Totales=DF2018.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2018Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2018Totales.rename(columns = {'freq':'Casos 2018'}, inplace = True)
shape=pd.merge(shape, DF2018Totales,how="outer",on='departamen')
shape['Casos 2018'] = shape['Casos 2018'].fillna(0)


DF2019=DF[DF['YEAR']==2019]
DF2019Totales=DF2019.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2019Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2019Totales.rename(columns = {'freq':'Casos 2019'}, inplace = True)
shape=pd.merge(shape, DF2019Totales,how="outer",on='departamen')
shape['Casos 2019'] = shape['Casos 2019'].fillna(0)


DF2020=DF[DF['YEAR']==2020]
DF2020Totales=DF2020.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2020Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2020Totales.rename(columns = {'freq':'Casos 2020'}, inplace = True)
shape=pd.merge(shape, DF2020Totales,how="outer",on='departamen')
shape['Casos 2020'] = shape['Casos 2020'].fillna(0)


DF2021=DF[DF['YEAR']==2021]
DF2021Totales=DF2021.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2021Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2021Totales.rename(columns = {'freq':'Casos 2021'}, inplace = True)
shape=pd.merge(shape, DF2021Totales,how="outer",on='departamen')
shape['Casos 2021'] = shape['Casos 2021'].fillna(0)


DF2022=DF[DF['YEAR']==2022]
DF2022Totales=DF2022.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2022Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2022Totales.rename(columns = {'freq':'Casos 2022'}, inplace = True)
shape=pd.merge(shape, DF2022Totales,how="outer",on='departamen')
shape['Casos 2022'] = shape['Casos 2022'].fillna(0)


DF2023=DF[DF['YEAR']==2023]
DF2023Totales=DF2023.groupby('DEPARTAMENTO_RESIDENCIA',as_index=False)['freq'].sum()
DF2023Totales.rename(columns = {'DEPARTAMENTO_RESIDENCIA':'departamen'}, inplace = True)
DF2023Totales.rename(columns = {'freq':'Casos 2023'}, inplace = True)
shape=pd.merge(shape, DF2023Totales,how="outer",on='departamen')
shape['Casos 2023'] = shape['Casos 2023'].fillna(0)


fig, ax = plt.subplots(figsize=(15, 15))
shape.plot(ax=ax,column='Casos 2023',legend=True)

# shape.plot()
ax.set_xlim(-75, -50)
ax.set_ylim(-55, -21)



for i in Provincias[Provincias=='Salta']:
    DFi=DF[DF['PROVINCIA_RESIDENCIA']==i]
    print(i)
    print(DFi['EDAD_DIAGNOSTICO'].describe())
    
    DFi.set_index('FECHA_NOTIFICACION', inplace=True)
    serieM=DFi[DFi['SEXO']=='M']['freq'].groupby('FECHA_NOTIFICACION').sum().resample('1M').sum()
    serieF=DFi[DFi['SEXO']=='F']['freq'].groupby('FECHA_NOTIFICACION').sum().resample('1M').sum()

    f, (ax1, ax2) = plt.subplots(1, 2, sharey=False)
    ax1.plot(serieM,'-*',color='blue',label='Male')
    ax1.plot(serieF,'-*',color='pink',label='Female')
    ax1.plot(serieM+serieF,'-*',color='black',label='Total')
    ax1.legend(loc='upper left')
    ax1.set_title('Reported Cases')
    
    ax2.plot(serieM.cumsum(),'',color='blue')
    ax2.plot(serieF.cumsum(),'',color='pink')
    ax2.plot(serieM.cumsum()+serieF.cumsum(),'',color='black')
    ax2.set_title('Total Cases')
    
    plt.suptitle(i)

    # sns.pairplot(DFi)

    
    # plt.Figure()
    # DFi.boxplot(column="EDAD_DIAGNOSTICO",by="SEXO")
    # plt.title(i)

# ============================================================================
#====================Mapa animado=============================================
# ============================================================================

# output_path='Mapa Animado/'

# fig, ax = plt.subplots(figsize=(15, 15))
# shape.plot(ax=ax,column='Casos 2023',legend=True)

# fechas=DF['FECHA_NOTIFICACION'].unique()
# DFAnimation = DF.drop_duplicates().groupby(['DEPARTAMENTO_RESIDENCIA','FECHA_NOTIFICACION'], sort=False, as_index=False).sum()

# DFAnimation = DFAnimation.rename(index=str, columns={"DEPARTAMENTO_RESIDENCIA": "departamen",
#                                                       "freq": "Casos"})

# DFAnimation=DFAnimation.drop(['EDAD_DIAGNOSTICO', 'YEAR'], axis=1)

# # merged1 = shape.set_index('departamen').join(DFAnimation.set_index('departamen'))

# # merged1['Casos']=merged1['Casos'].fillna(0)

# vmin=0
# vmax=DFAnimation['Casos'].max()
# variable='Casos'
# for i in fechas:
#     # i=fechas[360]
#     DFAnimationi=DFAnimation[DFAnimation['FECHA_NOTIFICACION']==i]
#     merged1 = shape.set_index('departamen').join(DFAnimationi.set_index('departamen'))

#     merged1['Casos']=merged1['Casos'].fillna(0)

#     # merged=merged1[merged1['FECHA_NOTIFICACION']==i]
#     # create map, UDPATE: added plt.Normalize to keep the legend range the same for all maps
#     fig = merged1.plot(column=variable,  figsize=(10,10),cmap='seismic', linewidth=1, edgecolor='0', vmin=vmin, vmax=vmax,
#                         legend=True,norm=plt.Normalize(vmin=vmin, vmax=vmax))
    
#     # remove axis of chart
#     fig.axis('off')
    
#     # add a title
#     fig.set_title('Casos diagosticados '+ str(i), \
#               fontdict={'fontsize': '25',
#                           'fontweight' : '3'})
    
#     # this will save the figure as a high-res png in the output path. you can also save as svg if you prefer.
#     filepath = os.path.join(output_path, str(i)+'Cases.jpg')
#     chart = fig.get_figure()
#     chart.savefig(filepath, dpi=300)
#     plt.close()
    

# import glob
# from PIL import Image

# # imagenes=list(os.listdir(output_path))
# imagenes0 = sorted( filter( os.path.isfile,glob.glob(output_path + '*') ) )

# imagenes=imagenes0
# image_list = [Image.open(file) for file in imagenes]

# image_list[0].save(
#             'animation10.gif',
#             save_all=True,
#             append_images=image_list[1:], # append rest of the images
#             duration=1000, # in milliseconds
#             loop=0)


# frame_folder=output_path
# def make_gif(frame_folder):
#     frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*")]
#     frame_one = frames[0]
#     frame_one.save("download.gif", format="GIF", append_images=frames,
#                 save_all=True, duration=1000, loop=1)
# if __name__=="__main__":
#     print("inside main")
#     make_gif(output_path)

# # =============================================================================
# # ==================================Video======================================
# # =============================================================================

# import moviepy.video.io.ImageSequenceClip
# image_folder=output_path
# fps=1
# imagenes0 = sorted( filter( os.path.isfile,glob.glob(output_path + '*') ) )
# imagenes0 = {x.replace(output_path, '')for x in imagenes0}
# imagenes0=sorted(imagenes0)
# while(output_path in imagenes0):
#     imagenes0.remove(output_path)

# image_files = [os.path.join(image_folder,img)
#                # for img in os.listdir(image_folder)
#                for img in imagenes0
#                if img.endswith(".jpg")]
# clip = moviepy.video.io.ImageSequenceClip.ImageSequenceClip(image_files, fps=fps)
# clip.write_videofile('Diamica.mp4')












    