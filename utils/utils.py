
import os
import pandas as pd
import xlrd
from unidecode import unidecode
import geopandas as gpd
import numpy as np
import pyreadstat

def formatear_dos_digitos(numero):
    numero=int(numero)
    if 0 <= numero < 10:
        return f"0{numero}"
    else:
        return str(numero)
    
def formatear_tres_digitos(numero):
    return str(numero).zfill(3)

def quitar_acentos_limpiar(s):  
    return(unidecode(str(s)).strip().replace("Villa ","").replace(" Del "," del ").replace(" De "," de ").replace(" La "," la ").replace(" Los "," los "))
    return(unidecode(str(s)).strip().title().replace("Villa ","").replace(" Del "," del ").replace(" De "," de "))

def antes_despues(tag,n1,n2,neg=0):
    print(tag+": ",n2,"de",n1,", un",(round((n2/n1)-neg,3))*100,"%")


def crear_provincas_argentina():
    yourpath = "Bases\\total_prov"
    df_provincias=pd.DataFrame()
    for root, dirs, files in os.walk(yourpath, topdown=False):
        for name in files:
            excel_file = os.path.join(root, name)

            workbook = xlrd.open_workbook(excel_file)
            sheet = workbook.sheet_by_index(0)
            data = []
            provincia = sheet.cell_value(0, 0).split(" (")[0].strip().title().replace("Del","del").replace("Ciudad Autonoma De Buenos Aires","CABA")

            numero = sheet.cell_value(0, 0).split(" (")[1].strip().replace(")","")
            for row in range(4,sheet.nrows):
                data.append(sheet.row_values(row))
            
            # Convert the data into a pandas DataFrame
            df_aux = pd.DataFrame(data)
            df_aux["Provincia"],df_aux["Codigo Provincia"]=provincia,numero

            df_provincias=pd.concat([df_provincias,df_aux])

    df_provincias=df_provincias.drop_duplicates()
    df_provincias.columns=["Departamento","Código","Localidad","CodEnt","Entidad comprendida","CodAglo","CodProv","CodDepto","CodLoc","Provincia","Codigo Provincia"]

    df_provincias=df_provincias.reset_index(drop=True)

    df_provincias.loc[df_provincias['Localidad']==""]
    df_provincias.head()
    localidad=""
    for i in df_provincias.index:
        if df_provincias.loc[i,"Localidad"]=="":
            df_provincias.loc[i,"Localidad"]=localidad

        if i==df_provincias.index[-1]:
            break  
        if df_provincias.loc[i,'Entidad comprendida']=="" and df_provincias.loc[i+1,'Entidad comprendida']=="":
            df_provincias.loc[i,'Entidad comprendida']=df_provincias.loc[i,"Localidad"]

        localidad=df_provincias.loc[i,"Localidad"]

    df_provincias.loc[df_provincias["Provincia"]=="Buenos Aires","Localidad"].unique()

    #df_provincias['Departamento'] = df_provincias['Departamento'].apply(lambda x: unidecode(x).strip()).str.strip().str.replace("Villa ","")
    df_provincias['Provincia'] = df_provincias['Provincia'].apply(quitar_acentos_limpiar)
    i=1
    for s in df_provincias.loc[df_provincias['Provincia'].str.title()=="Caba","Departamento"]:
        df_provincias.loc[(df_provincias['Provincia'].str.title()=="Caba")&(df_provincias["Departamento"]==s),"Departamento"]="Comuna "+str(i)
        i+=1

    df_provincias["Departamento"] = df_provincias["Departamento"].apply(quitar_acentos_limpiar)
    df_provincias['Localidad'] = df_provincias['Localidad'].apply(quitar_acentos_limpiar)
    df_provincias['Entidad comprendida'] = df_provincias['Entidad comprendida'].apply(quitar_acentos_limpiar)
    df_provincias.loc[df_provincias["CodDepto"]=="","CodDepto"]=0
    df_provincias["CodDepto"]=df_provincias["CodDepto"].astype(int)
    df_provincias["CodDepto"]=df_provincias["CodDepto"].apply(lambda x: formatear_tres_digitos(x))

    df_provincias["CodProvDepto"]=df_provincias["Codigo Provincia"].astype(str)+df_provincias["CodDepto"]
    df_provincias["Departamento"]=df_provincias["Departamento"].apply(quitar_acentos_limpiar)
    df_provincias["ID_DEPTO_INDEC_RESIDENCIA"]=df_provincias["CodDepto"]
    df_provincias.to_csv("Bases/provincias_total.csv",index=False)
    return(df_provincias)



import chardet

def crear_pobreza(df_provincias):

    with open("Bases\\pobreza_otros.csv", 'rb') as file:
        result = chardet.detect(file.read(10000))  # Lee una muestra del archivo para detectar la codificación

    encoding = result['encoding']
    print(f"Detected encoding: {encoding}")

    # Leer el archivo con la codificación detectada
    df_pobreza = pd.read_csv("Bases\\pobreza_otros.csv", sep="\t",encoding=encoding)

    # Verificar las primeras filas del DataFrame para asegurarte de que se ha leído correctamente
    df_pobreza.head()
    df_pobreza=df_pobreza.rename(columns={"Provincia con error AGBA":"Provincia"})

    df_pobreza["Provincia"]=df_pobreza["Provincia"].str.replace('Partidos del AGBA',"Buenos Aires").str.replace('Ciudad de Buenos Aires',"CABA").apply(quitar_acentos_limpiar)

    df_pobreza["Departamento"]=df_pobreza["Departamento"].str.replace("Comuna 0","Comuna ").str.replace("  "," ").apply(quitar_acentos_limpiar)

    df_provincias2=df_provincias.drop_duplicates(["Departamento","Provincia"])
    df_provincias2["Departamento"]=df_provincias2["Departamento"].str.replace("Grl. ","General ").str.replace("l Juan F.","l Juan Facundo")
    df_pobreza_codigo=pd.merge(df_provincias2,df_pobreza,on=["Provincia","Departamento"],how="left")

    df_pobreza_codigo["link"]=df_pobreza_codigo["Código"].str.replace(".0","").str[:-3]
 
    df_pobreza_codigo=df_pobreza_codigo[['Nivel de incidencia de pobreza crónica','% de hogares con hacinamiento crítico',"% de hogares con jefe/a con primario completo o menos","% de hogares con jefe/a con secundario incompleto o menos","% de hogares en vivienda deficitaria","% de hogares sin acceso a red cloacal","% de niños/as entre 6 y 17 años que no asisten a la escuela","% de población en situación de pobreza crónica","% de población sin obra social ni prepaga","% de población urbana","Población","link"]]

    df_pobreza_codigo=df_pobreza_codigo.replace(',', '.', regex=True)
    df_pobreza_codigo.to_csv("Bases\\pobreza.csv",index=False)


    return(df_pobreza_codigo)

def poblacion_arg():
    lista_archivos=["densidad_2010","densidad_2022","poblacion_2010","poblacion_2022","salud_17_2022","salud_64_2022","salud_65_2022"]
    df_final_pob=pd.DataFrame()
    df_final_pob["provincia"]=["","",""]
    for f in lista_archivos:

        fname = "Bases/Poblacionales/%s.geojson" % (f)

        df_aux = gpd.read_file(fname,encoding="latin1")[["nam","value"]]
        df_aux.columns=["provincia",f]
        df_final_pob=pd.merge(df_final_pob,df_aux,on=["provincia"],how="right")
    df_final_pob.loc[df_final_pob["provincia"]=='Rí\xado Negro',"provincia"]="Río Negro"
    df_final_pob.head()
    df_final_pob2=df_final_pob.copy()

    df_final_pob2["provincia"]=df_final_pob2["provincia"].str.replace("Ciudad Autónoma de Buenos Aires","CABA")

    df_final_pob2.loc[df_final_pob2["provincia"]=='Tierra del Fuego, Antártida e Islas del Atlántico Sur',"Provincia"]='Tierra del Fuego'
    df_final_pob2["provincia"]=df_final_pob2["provincia"].apply(quitar_acentos_limpiar)

    df_final_pob2.to_csv("Bases\\poblacion_arg.csv",index=False)
    return(df_final_pob2)


def obesity_data():

    path_name='Bases\\Conicet\\CONICET_Digital_Nro.faadb325-d9c6-4df6-b361-99befc65e702_A.dta'

    df2_ob, _ = pyreadstat.read_dta(path_name)
    df2_ob.head()

    for c in df2_ob.columns:
        if ("AdjOb_" in c and "SEA" not in c) or c=="cod_prov" or c=="pcia":
            pass
        else:
            df2_ob.drop(c,axis=1,inplace=True)


    df2_ob_m=df2_ob.copy()[['cod_prov','pcia','AdjOb_Masc05','AdjOb_Masc09','AdjOb_Masc13','AdjOb_Masc18']]
    df2_ob_f=df2_ob.copy()[['cod_prov','pcia','AdjOb_Fem05','AdjOb_Fem09','AdjOb_Fem13','AdjOb_Fem18']]


    # Extracting years from column names
    years = [int(column.split('_')[-1][-2:]) for column in df2_ob_m.columns if 'AdjOb_Masc' in column]

    # Create a new DataFrame with years as columns
    new_years = [f'{i}' for i in range(2005, 2019)]
    new_df = pd.DataFrame(columns=['cod_prov', 'pcia'] + new_years)

    # Iterate over the original DataFrame to populate the new DataFrame
    for index, row in df2_ob_m.iterrows():
        cod_prov = row['cod_prov']
        pcia = row['pcia']
        values = [row[column] for column in df2_ob_m.columns if 'AdjOb_Masc' in column]
        
        # Check if there are available values for interpolation
        if all(np.isnan(values)):
            # If all values are NaN, skip interpolation
            interpolated_values = [np.nan] * len(new_years)
        else:
            # Interpolate values between 2005 and 2009
            interpolated_values = np.interp(np.arange(5, 19), years, values)
        
        new_df.loc[index] = [cod_prov, pcia] + list(interpolated_values)

    # Set the new DataFrame as df2_ob_m
    df2_ob_m = new_df
    df2_ob_m2=pd.melt(df2_ob_m,id_vars=["cod_prov","pcia"],var_name="fnyear",value_name="index obesity")
    df2_ob_m2["sexo"]="M"

    for index, row in df2_ob_f.iterrows():
        cod_prov = row['cod_prov']
        pcia = row['pcia']
        values = [row[column] for column in df2_ob_f.columns if 'AdjOb_Fem' in column]
        
        # Check if there are available values for interpolation
        if all(np.isnan(values)):
            # If all values are NaN, skip interpolation
            interpolated_values = [np.nan] * len(new_years)
        else:
            # Interpolate values between 2005 and 2009
            interpolated_values = np.interp(np.arange(5, 19), years, values)
        
        new_df.loc[index] = [cod_prov, pcia] + list(interpolated_values)

    # Set the new DataFrame as df2_ob_m
    df2_ob_f = new_df
    df2_ob_f2=pd.melt(df2_ob_f,id_vars=["cod_prov","pcia"],var_name="fnyear",value_name="index obesity")

    df2_ob_f2["sexo"]="F"
    df3_ob=pd.concat([df2_ob_f2,df2_ob_m2])
    # Crear una lista para almacenar las filas proyectadas
    projected_rows = []
    additional_years=[2019,2020,2021,2022]
    # Iterar sobre los datos interpolados para cada combinación de región, sexo y año

    df3_ob2018=df3_ob.loc[df3_ob["fnyear"]=="2018"]
    for index, row in df3_ob2018.iterrows():
        cod_prov = row['cod_prov']
        pcia = row['pcia']
        index_obesity = row["index obesity"]
        sexo = row['sexo']
        
        # Calcular el crecimiento anual promedio
        growth_rate = (index_obesity / (2018 - 2005))  # Promedio de crecimiento entre 2005 y 2018
        
        # Proyectar valores para los años adicionales
        for additional_year in additional_years:
            projected_index_obesity = index_obesity + growth_rate * (int(additional_year) - 2018)
            
            # Agregar la fila proyectada a la lista
            projected_rows.append({'cod_prov': cod_prov,
                                'pcia': pcia,
                                'fnyear': additional_year,
                                "index obesity": projected_index_obesity,
                                'sexo': sexo})

    # Crear un DataFrame a partir de la lista de filas proyectadas
    projected_df = pd.DataFrame(projected_rows)
    projected_df2=pd.concat([df3_ob,projected_df])

    projected_df2["fnyear"]=projected_df2["fnyear"].astype(str)
    projected_df2.to_csv("Bases\\obesity_data.csv",index=False)
    return(projected_df2)

def kelvin_to_celsius(k):
    return k - 273.15+33.15
def interpolate_group(group):
    group['Relative humidity'] = group['Relative humidity'].interpolate()
    group['Specific humidity'] = group['Specific humidity'].interpolate()
    group['Temperature'] = group['Temperature'].interpolate()
    group['Temperature2'] = group['Temperature2'].interpolate()
    return group

def full_climage(clima_provincias):


    clima_provincias['valid_date'] = pd.to_datetime(clima_provincias['valid_date'])

    date_range = pd.date_range(start='2009-01-01', end='2022-12-31', freq='D')


    provincias_departamentos = clima_provincias[['provincia', 'departamen', 'link']].drop_duplicates()
    provincias_departamentos['key'] = 1
    dates = pd.DataFrame(date_range, columns=['valid_date'])
    dates['key'] = 1


    all_combinations = pd.merge(provincias_departamentos, dates, on='key').drop('key', axis=1)


    all_combinations['valid_date'] = pd.to_datetime(all_combinations['valid_date'])


    clima_provincias_full = pd.merge(all_combinations, clima_provincias, on=['provincia', 'departamen', 'link', 'valid_date'], how='left')


    clima_provincias_full = clima_provincias_full.sort_values(by=['provincia', 'departamen', 'link', 'valid_date'])




    clima_provincias_full = clima_provincias_full.groupby(['provincia', 'departamen', 'link']).apply(interpolate_group)

    clima_provincias_full = clima_provincias_full.reset_index(drop=True)
    return(clima_provincias_full)
def climatage_data():
    ruta_archivo_shp = 'Bases\\\Vecindad\\Codgeo_Pais_x_dpto_con_datos\\pxdptodatosok.shp'

    dataframe_shp = gpd.read_file(ruta_archivo_shp)

    dataframe_shp["provincia"]=dataframe_shp["provincia"].apply(utils.quitar_acentos_limpiar).str.replace("Ciudad Autonoma de Buenos Aires","Caba")

    dataframe_shp["departamen"]=dataframe_shp["departamen"].apply(utils.quitar_acentos_limpiar)

    output_dir = "Bases\\Climatologicos"

    dataframes = []

    files = [f for f in os.listdir(output_dir) if f.startswith("temp_") and f.endswith(".csv")]

    for file in files:
        file_path = os.path.join(output_dir, file)
        df = pd.read_csv(file_path)
        dataframes.append(df)
        print(f"Leído {file_path}")

    df_grib = pd.concat(dataframes, ignore_index=True)

    df_grib.loc[df_grib["message_name"]=="Temperature"].describe()

    geometry = [Point(xy) for xy in zip(df_grib['longitude'], df_grib['latitude'])]
    dataframe_shp2=dataframe_shp[["codpcia","departamen","provincia","geometry","link"]]
    df_grib= gpd.GeoDataFrame(df_grib, geometry=geometry)

    dataframe_shp2 = gpd.GeoDataFrame(dataframe_shp2, geometry='geometry')
    df_grib2 = gpd.GeoDataFrame(df_grib, geometry='geometry')

    dataframe_shp2 = dataframe_shp2.set_crs('EPSG:4326')
    df_grib2 = df_grib2.set_crs('EPSG:4326')

    df_grib2['centroid'] = df_grib2['geometry'].centroid

    gdf_grib_points = gpd.GeoDataFrame(df_grib2, geometry='centroid')

    points_within_polygons = gpd.sjoin(gdf_grib_points, dataframe_shp2, how='inner', op='within')

    clima_provincias1=pd.DataFrame(pd.pivot_table(points_within_polygons,index=["provincia","departamen","link","valid_date","message_name"],values="value",aggfunc="mean").to_records())

    dataframe_shp_not=dataframe_shp2.loc[~dataframe_shp2["link"].isin(list(points_within_polygons["link"]))]

    dataframe_shp_not['centroid'] = dataframe_shp_not['geometry'].centroid

    dataframe_shp_not['centroid_lat'] = dataframe_shp_not['centroid'].apply(lambda x: x.y)
    dataframe_shp_not['centroid_lon'] = dataframe_shp_not['centroid'].apply(lambda x: x.x)


    df_centroids = dataframe_shp_not[["provincia","link",'departamen', 'centroid_lat', 'centroid_lon']].copy()

    points = df_grib[['latitude', 'longitude']].values
    values = df_grib['value'].values

    centroid_points = df_centroids[['centroid_lat', 'centroid_lon']].values

    # centroid_values = griddata(points, values, centroid_points, method='linear')

    tree = cKDTree(points)

    _, indices = tree.query(centroid_points)

    # centroid_values = values[indices]

    df_centroids['latitude'] = df_grib2.loc[indices, 'latitude'].values
    df_centroids['longitude'] = df_grib2.loc[indices, 'longitude'].values

    df_centroids2=pd.merge(df_centroids,df_grib2,on=["latitude","longitude"],how="left")

    clima_provincias2=df_centroids2.drop(["geometry","centroid","centroid_lat","centroid_lon","latitude","longitude"],axis=1)
    clima_provincias2

    clima_provincias_f=pd.concat([clima_provincias1,clima_provincias2])

    clima_provincias_f2 = clima_provincias_f.pivot_table(index=['provincia', 'departamen', 'link', 'valid_date'],
                            columns='message_name',
                            values='value',
                            aggfunc='first').reset_index()

    # Aplicar la función a la columna 'Temperature'
    clima_provincias_f2['Temperature2'] = clima_provincias_f2['Temperature'].apply(kelvin_to_celsius)

    clima_provincias_f2.to_csv("Bases\\climate_data.csv",index=False)
    clima_provincias_full=full_climage(clima_provincias_f2)
    clima_provincias_full.to_csv("Bases\\climate_data_full.csv",index=False)
    return(clima_provincias_f2)