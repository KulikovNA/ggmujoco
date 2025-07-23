import pathlib

def gen_manip_block(path_xml, assets):
    path_xml = pathlib.Path(path_xml).resolve()
    assets = pathlib.Path(assets).resolve()
    return f'''
        <include file="{path_xml.resolve()}"/>
      <compiler angle="radian"
        meshdir="{assets.resolve()}"/>'''