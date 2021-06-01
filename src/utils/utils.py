# import geopandas as gpd
import pandas as pd
# from shapely.geometry import Point

# def convert_gpd(station_info, lat_col, lon_col, crs=4269):
#     """convert station info Pandas DataFrame into GeoPandas DataFrame

#     Args:
#         station_info (pd.DataFrame): pandas station information DataFrame
#         lat_col (string): latitude column name
#         lon_col (string): longitude column name

#     Returns:
#         gpd.GeoDataFrame: GeoPandas DataFrame of station info
#     """    
#     df = station_info.copy()
#     get_geometry = lambda x, lon, lat: Point(x[lon], x[lat])
#     df['geometry'] = df.apply(lambda x: get_geometry(x, lon_col, lat_col), axis = 1)

#     return gpd.GeoDataFrame(df, crs={'init':f'epsg:{crs}'}, \
#                             geometry='geometry')
def split_sequences(sequences):
    """split sequence into sequences of continues list

    Args:
        sequences (list/np.array): the array / list contains one or more continues sequences

    Returns:
        list: list of array with elements of continues sequences [<array of continues seq>, <array of continues seq>, ...]
    """    
    break_points = [0]
    for idx, (s, e) in enumerate(zip(sequences[:-1], sequences[1:])):
        if e - s > 1:
            break_points.append(idx+1)
    break_points.append(len(sequences) - 1)
    break_points = list(zip(break_points[:-1], break_points[1:]))
    splitted_seq = sequences = [sequences[s: e] for s, e in break_points]
    return splitted_seq