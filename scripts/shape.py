
import pandas as pd 
import geopandas as gpd
import os
import sys
import copy 

def load_basin_shape(shape_fpath: str,
                     shape_crs: str,
                     aso_site_name: str = None,
                    buffer_size: int = 10000
                    ):
    """
    Loads in basin shape based on either shapefile or basin name.
    Input:
      shape_fpath - python string object or Nonetype object for relative filepath 
                    of basin shapefile. If None provided then its assumed the 
                    basin string is provided.
      shape_crs - python string object of epsg string for projection.
                        - e.g. 'EPSG:32611'
      aso_site_name - python string object of aso site name.
                        - e.g. 'USCATM'
                        - https://nsidc.org/sites/default/files/aso_basins.pdf
    Output:
      aso_gdf_geog - geopandas object of aso shape in lat / long coordinates
      aso_gdf_proj - geopandas object of aso shape in projected coordinates
      aso_buff_gdf_geog - geopandas object of aso buffered shape in lat / long coordinates
      aso_buff_gdf_proj - geopandas object of aso buffered shape in projected coordinates
    """
    
    ## exit if no shape information is provided ##
    if (shape_fpath is None) and (aso_site_name is None):
        print('Need to provide either basin shapefile or basin site code as'\
              'provided on https://nsidc.org/sites/default/files/aso_basins.pdf')
        sys.exit()

    ## if shapefile is provided ##   
    if (shape_fpath is not None):  
        if os.path.isfile(shape_fpath):
            aso_gdf_proj = gpd.read_file(shape_fpath).to_crs(shape_crs)
        else:
            print('Incorrect filepath to shapefile')
            sys.exit()
        
    ## if name is passed ## NEED TO FILL IN THIS PART!!
    else:
        aso_gdf_proj = load_aso_basin_table().to_crs(shape_crs)
        basin_codes =  aso_gdf_proj['site_code'].to_list()
        
        ## check to see if provided site code is NOT in list ##
        if aso_site_name not in (basin_codes):
            print(f'Incorrect site code provided. Choose from following'\
                  f'list {basin_codes}')
            sys.exit()
        ## select aso basin ## 
        aso_gdf_proj = aso_gdf_proj[aso_gdf_proj['site_code'] == aso_site_name]
    
    ## create lat lon geopandas ##
    aso_gdf_geog = aso_gdf_proj.to_crs('EPSG:4326')
    
    ## create buffered geopandas ##
    aso_gdf_proj_buff = copy.deepcopy(aso_gdf_proj)
    aso_gdf_proj_buff['buff_geom'] = aso_gdf_proj_buff.geometry.buffer(buffer_size)
    aso_gdf_proj_buff.drop(columns = ['geometry'],inplace = True)
    aso_gdf_proj_buff.rename(columns = {'buff_geom':'geometry'},inplace = True)

    aso_gdf_geog_buff = aso_gdf_proj_buff.to_crs('EPSG:4326')
    
    return aso_gdf_geog,aso_gdf_proj,aso_gdf_geog_buff,aso_gdf_proj_buff