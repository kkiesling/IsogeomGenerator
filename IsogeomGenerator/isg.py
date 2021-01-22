import sys
import os
import shutil
import warnings
import numpy as np
import math as m

from isg_gen import IsoGeomGen

from pymoab import core, types
from pymoab.rng import Range
from pymoab.skinner import Skinner


class IsGm(IsoGeomGen):
    """Class containing necessary methods for generating a full mesh
    geometry from a database of isosurface volumes. This class uses the
    python interface for MOAB (pyMOAB).

    Attributes:
    -----------
        levels: list of floats, values used for isosurface values
        data: string, name of data on mesh
        db: string, path to database folder with isovolume files
        mb: MOAB core instance, contains all information of the mesh
            geometry
        isovol_meshsets: dictionary, information relating curve,
            surface, and volume entity handles to each other.
        surf_curve: dictionary, information relating surfaces to their
            curves.  key = surf eh, value = list of child curve eh
        val_tag: MOAB tag entity handle, tag for surface value
        sense_tag: MOAB tag entity handle, tag for surface sense

    Methods:
    --------
    """

    def __init__(self, ivdb=None, levels=None, data=None, db=None):
        """Create IsGm object. Information provided by an ivdb object
        will overwrite other data provided.

        Input:
        ------
            ivdb: (object), IvDb object, must be a completed object
            levels: (optional), list of floats or string, values to use
                for isosurface values. If string, then it is the path
                to levelfile where one value is given per line.
            data: (optional), string, name of data on the mesh
            db: (optional), string, path to database folder with
                isovolume files
        """
        # initialize variables
        super(IsGm, self).__init__(levels, data, db)

        # if ivdb object is provided, overwrite with that info
        if ivdb is not None:
            self.read_ivdb(ivdb)

        # set MOAB related attributes
        self.mb = core.Core()
        self.isovol_meshsets = {}
        self.surf_curve = {}
        self.val_tag = \
            self.mb.tag_get_handle(self.data, size=1,
                                   tag_type=types.MB_TYPE_DOUBLE,
                                   storage_type=types.MB_TAG_SPARSE,
                                   create_if_missing=True)
        self.sense_tag = \
            self.mb.tag_get_handle('GEOM_SENSE_2', size=2,
                                   tag_type=types.MB_TYPE_HANDLE,
                                   storage_type=types.MB_TAG_SPARSE,
                                   create_if_missing=True)

    def read_ivdb(self, ivdb):
        """read information from IvDb object.

        Input:
        ------
            ivdb: completed IvDb object
        """
        # check that object is completed
        if not ivdb.completed:
            raise RuntimeError("Incomplete IvDb object was " +
                               "provided. Please run 'generate_volumes()'.")
        # set values
        self.levels = ivdb.levels
        self.db = ivdb.db
        self.data = ivdb.data

    def read_database(self):
        """Read the files from the database and initialize the meshset info.
        """
        # check that levels exist:
        if self.levels is None:
            raise RuntimeError("Object must have levels defined.")

        # check there are correct number of files:
        file_list = sorted(os.listdir(self.db + "/vols/"))
        if len(self.levels) + 1 != len(file_list):
            raise RuntimeError("Number of levels does not match number of " +
                               "isovolume files in the database.")

        # read files
        for f in file_list:
            # get file name
            fpath = self.db + "/vols/" + f
            i = int(f.strip(".stl"))  # must be an integer

            # load file and create EH for file-set
            fs = self.mb.create_meshset()
            self.mb.load_file(fpath, file_set=fs)

            # initiate dictionary
            iv_info = (i, fs)
            self.isovol_meshsets[iv_info] = {}

            # add value min/max info (min, max)
            if i == 0:
                self.isovol_meshsets[iv_info]['bounds'] =\
                    (None, self.levels[i])
            elif i == len(self.levels):
                self.isovol_meshsets[iv_info]['bounds'] =\
                    (self.levels[i - 1], None)
            else:
                self.isovol_meshsets[iv_info]['bounds'] =\
                    (self.levels[i - 1], self.levels[i])

    def separate_isovols(self):
        """For each isovolume in the database, separate any disjoint
        surfaces into unique single surfaces.
        """
        for iv_info in self.isovol_meshsets.keys():
            # extract isovolume information
            iso_id = iv_info[0]
            fs = iv_info[1]

            # get set of all vertices for the isosurface
            all_verts = self.mb.get_entities_by_type(fs, types.MBVERTEX)

            # initiate list to store separate surface entity handles
            self.isovol_meshsets[iv_info]['surfs_EH'] = []

            # separate the surfaces
            print("Separating isovolume {}".format(iso_id))
            while len(all_verts) > 0:
                # get full set of connected verts starting from a seed
                verts = [all_verts[0]]
                verts_check = [all_verts[0]]
                vtmp_all = set(verts[:])

                # gather set of all vertices that are connected to the seed
                while True:
                    # check adjancency and connectedness of new vertices
                    vtmp = self.mb.get_adjacencies(self.mb.get_adjacencies(
                                                   verts_check, 2, op_type=1),
                                                   0, op_type=1)

                    # add newly found verts to all list
                    vtmp_all.update(set(vtmp))

                    # check if different from already found verts
                    if len(list(vtmp_all)) == len(verts):
                        # no more vertices are connected, so full surface
                        # has been found
                        break
                    else:
                        # update vertices list to check only newly found
                        # vertices
                        verts_check = vtmp_all.difference(verts)
                        verts = list(vtmp_all)

                # get the connected set of triangles that make the single
                # surface and store into a unique meshset
                tris = self.__get_surf_triangles(verts)
                surf = self.mb.create_meshset()
                self.mb.add_entities(surf, tris)
                self.mb.add_entities(surf, verts)

                # store surfaces in completed list
                self.isovol_meshsets[iv_info]['surfs_EH'].append(surf)

                # remove surface from original meshset
                self.mb.remove_entities(fs, tris)
                self.mb.remove_entities(fs, verts)

                # resassign vertices that remain
                all_verts = self.mb.get_entities_by_type(fs, types.MBVERTEX)

    def imprint_merge(self, norm, merge_tol):
        """Uses PyMOAB to check if surfaces are coincident. Creates a
        single surface where surfaces are coincident values are tagged
        on each surface. Surface senses are also determined and tagged.

        Input:
        ------
            norm: float, All data values will be multiplied by this factor.
            merge_tol: float, Merge tolerance for mesh based merge of
                coincident surfaces.
        """
        # get list of all original isovolumes
        all_vols = sorted(self.isovol_meshsets.keys())
        for i, isovol in enumerate(all_vols):
            if i != len(self.levels):
                # do not need to check the last isovolume because it
                # will be checked against its neighbor already
                self.__compare_surfs(isovol, all_vols[i + 1], norm, merge_tol)

        # if a surface doesn't have a value tagged after merging
        # give it a value of 0 and tag forward sense
        for isovol in all_vols:
            for surf in self.isovol_meshsets[isovol]['surfs_EH']:

                # tag val=0
                try:
                    val = self.mb.tag_get_data(self.val_tag, surf)
                except:
                    val = 0.0
                    self.mb.tag_set_data(self.val_tag, surf, val)
                    verts = \
                        self.mb.get_entities_by_type(surf,
                                                     types.MBVERTEX)
                    tris = self.__get_surf_triangles(verts)
                    self.mb.add_entities(surf, tris)

                # tag fwd sense
                try:
                    sense = self.mb.tag_get_data(self.sense_tag, surf)
                except:
                    fwd = isovol[1]
                    bwd = np.uint64(0)
                    self.mb.tag_set_data(self.sense_tag,
                                         surf, [fwd, bwd])

    def make_family(self):
        """Makes the correct parent-child relationships with volumes
        and surfaces. Tags geometry type, category, and ID on surfaces
        and volumes.
        """
        # create geometry dimension, category, and global id tags
        geom_dim = \
            self.mb.tag_get_handle('GEOM_DIMENSION', size=1,
                                   tag_type=types.MB_TYPE_INTEGER,
                                   storage_type=types.MB_TAG_SPARSE,
                                   create_if_missing=True)
        category = \
            self.mb.tag_get_handle('CATEGORY', size=32,
                                   tag_type=types.MB_TYPE_OPAQUE,
                                   storage_type=types.MB_TAG_SPARSE,
                                   create_if_missing=True)
        global_id = \
            self.mb.tag_get_handle('GLOBAL_ID', size=1,
                                   tag_type=types.MB_TYPE_INTEGER,
                                   storage_type=types.MB_TAG_SPARSE,
                                   create_if_missing=True)

        vol_id = 0
        surf_id = 0
        completed_surf_list = []
        for v in self.isovol_meshsets.keys():
            vol_eh = v[1]

            # tag volume
            self.mb.tag_set_data(geom_dim, vol_eh, 3)
            self.mb.tag_set_data(category, vol_eh, 'Volume')
            vol_id += 1
            self.mb.tag_set_data(global_id, vol_eh, vol_id)

            for surf_eh in self.isovol_meshsets[v]['surfs_EH']:
                # create relationship
                self.mb.add_parent_child(vol_eh, surf_eh)

                # tag surfaces (if not already done)
                if surf_eh not in completed_surf_list:
                    self.mb.tag_set_data(geom_dim, surf_eh, 2)
                    self.mb.tag_set_data(category, surf_eh, 'Surface')
                    surf_id += 1
                    self.mb.tag_set_data(global_id, surf_eh, surf_id)
                    completed_surf_list.append(surf_eh)

        curve_id = 0
        completed_curve_list = []
        for s in self.surf_curve.keys():
            for c in self.surf_curve[s]:
                # create relationship
                self.mb.add_parent_child(s, c)

                # tag curves (if not already done)
                if c not in completed_curve_list:
                    self.mb.tag_set_data(geom_dim, c, 1)
                    self.mb.tag_set_data(category, c, 'Curve')
                    curve_id += 1
                    self.mb.tag_set_data(global_id, c, curve_id)
                    completed_curve_list.append(c)

    def tag_for_viz(self):
        """Tags all triangles on all surfaces with the data value for
        that surface. This is for vizualization purposes.
        """
        for isovol in self.isovol_meshsets.keys():
            for surf in self.isovol_meshsets[isovol]['surfs_EH']:
                # get the tagged data
                val = self.mb.tag_get_data(self.val_tag, surf)

                # get the triangles
                tris = self.mb.get_entities_by_type(surf,
                                                    types.MBTRI)

                # create data array
                num = len(tris)
                data = np.full((num), val)

                # tag the data
                self.mb.tag_set_data(self.val_tag, tris, data)

    def set_tags(self, tags):
        """Set provided tag values on the root set.

        Tag size and type will be determined by the TAGVALUE. Any
        TAGNAME that is not a string will be converted to a string.

        Input:
        ------
            tags: dict, key=TAGNAME, value=TAGVALUE, TAGNAMEs should be
                strings and TAGVALUEs can be single values or lists of
                any type.
        """
        rs = self.mb.get_root_set()
        for tagname, tagval in tags.items():
            # get tag size
            try:
                taglength = len(tagval)  # strings or lists of ints/floats
            except:
                taglength = 1  # can't get length on a single int/float values

            # get datatype:
            if taglength > 1:
                mbtype = types.pymoab_data_type(type(tagval[0]))
            else:
                mbtype = types.pymoab_data_type(type(tagval))

            # create tag
            tag = self.mb.tag_get_handle(str(tagname), size=taglength,
                                         tag_type=mbtype,
                                         storage_type=types.MB_TAG_SPARSE,
                                         create_if_missing=True)
            self.mb.tag_set_data(tag, rs, tagval)

    def write_geometry(self, sname, sdir):
        """Writes out the geometry stored in memory.

        Input:
        ------
            sname: string, name of file to save written file
            sdir: string, absolute path for writing file
        """
        # check file extension of save name:
        ext = sname.split(".")[-1]
        if ext.lower() not in ['h5m', 'vtk']:
            warnings.warn("File extension {} ".format(ext) +
                          " not recognized. File will be saved as type .h5m.")
            sname = sname.split(".")[0] + ".h5m"
        # save the file
        save_location = sdir + "/" + sname
        self.mb.write_file(save_location)
        print("Geometry file written to {}.".format(save_location))

    def __get_surf_triangles(self, verts_good):
        """This function will take a set of vertex entity handles and
        return the set of triangles for which all vertices of all
        triangles are in the set of vertices.

        Input:
        ------
            verts_good: list of entity handles, list of vertices to
                compare against. Only triangles will be returned whose
                complete set of vertices are in this list.

        Returns:
        --------
            tris: list of entity handles, EHs for the triangles for
                which all three vertices are in the verts list.
        """
        tris_all = self.mb.get_adjacencies(verts_good, 2, op_type=1)
        verts_all = self.mb.get_connectivity(tris_all)
        verts_bad = set(verts_all) - set(verts_good)

        if verts_bad:
            # not an empty set
            tris_bad = self.mb.get_adjacencies(list(verts_bad), 2, op_type=1)
            tris_good = set(tris_all) - set(tris_bad)
            return list(tris_good)
        else:
            # empty set so all tris are good
            return tris_all

    def __list_coords(self, eh, invert=False):
        """Gets list of all coords as a list of tuples for an entity
        handle eh.

        Input:
        ------
            eh: MOAB entity handle for meshset to retrieve coordinates
            invert: bool, default=False, True to invert keys and values
                in returned coords dict

        Returns:
        --------
            coords: dictionary, key is the MOAB entity handle for the
                vertice and the value is a tuple of the coordinate
                (x, y, z). If invert=True, keys and values are switched.
        """
        # list of all entity handles for all vertices
        all_verts_eh = self.mb.get_entities_by_type(eh, types.MBVERTEX)
        coords = {}
        for v in all_verts_eh:
            coord = tuple(self.mb.get_coords(v))

            if invert:
                # invert is true
                key = coord
                value = v
            else:
                key = v
                value = coord

            coords[key] = value

        return coords

    def __get_matches(self, vertsA, vertsB, merge_tol):
        """Collects the set of entity handles and coordinates in set of
        vertsA and vertsB that match within the specified absolute
        tolerance (merge_tol).

        Input:
        ------
            vertsA: dictionary, key is the MOAB entity handle for the
                vertex and the value is a tuple of the coordinate
                (x, y, z)
            vertsB: dictionary, key is a tuple of the coordinate and the
                value is the MOAB entity handle for the coordinate.
            merge_tol: float, Merge tolerance for mesh based merge of
                coincident surfaces.

        Returns:
        --------
            sA_match_eh: list of MOAB entity handles, the entity handles
                for set vertsA that exist is vertsB
            sA_match_coords: list of tuples, each entry is the
                corresponding coordinate for the EH in sA_match_eh
            sB_match_eh: list of MOAB entity handles, the entity handles
                for set vertsB that exist is vertsA
            sB_match_coords: list of tuples, each entry is the
                corresponding coordinate for the EH in sB_match_eh
        """
        sA_match_eh = []
        sA_match_coords = []
        sB_match_eh = []
        sB_match_coords = []
        match_dict = {}
        bcoords = vertsB.keys()

        # get exact matches
        for vert in vertsA.items():
            ehA = vert[0]
            coord = vert[1]
            if coord in bcoords:
                # exact match
                sA_match_eh.append(ehA)
                sA_match_coords.append(coord)
                sB_match_coords.append(coord)
                sB_match_eh.append(vertsB[coord])

                match_dict[vertsB[coord]] = ehA

            else:
                # check approx
                tf = np.isclose(coord, bcoords, rtol=0, atol=merge_tol)

                # get index of matches if they exist
                ix = np.where(zip(*tf)[0])[0]
                iy = np.where(zip(*tf)[1])[0]
                iz = np.where(zip(*tf)[2])[0]

                # get index if only x y and z match
                index_set = list(set(ix) & set(iy) & set(iz))

                if index_set != []:
                    index = index_set[0]

                    # get the close match coordinate in the bcoords list
                    bcoord = bcoords[index]
                    ehB = vertsB[bcoord]

                    sA_match_eh.append(ehA)
                    sA_match_coords.append(coord)
                    sB_match_eh.append(ehB)
                    sB_match_coords.append(bcoord)

                    match_dict[ehB] = ehA

        return sA_match_eh, sA_match_coords, sB_match_eh, sB_match_coords, \
            match_dict

    def __compare_surfs(self, v1, v2, norm, merge_tol):
        """finds coincident surfaces between two isovolumes.

        Input:
        ------
            v1/2: tuple, corresponds to the dictionary keys for two
                isovolumes in self.isovol_meshsets that will be compared
            norm: float, All data values will be multiplied by this factor.
            merge_tol: float, Merge tolerance for mesh based merge of
                coincident surfaces.
        """
        print("comparing surfaces in isovolumes {} and {}.".format(
            v1[0], v2[0]))

        match_surfs = []
        sk = Skinner(self.mb)

        # compare all surfaces in v1 (s1) to all surfaces in v2 (s2)
        for s1 in self.isovol_meshsets[v1]['surfs_EH']:
            # get list of all coordinates in s1
            verts1 = self.__list_coords(s1)

            # initialize list of curves
            if s1 not in self.surf_curve.keys():
                self.surf_curve[s1] = []

            for s2 in self.isovol_meshsets[v2]['surfs_EH']:
                if s2 not in self.surf_curve.keys():
                    self.surf_curve[s2] = []

                # get list of all coordinates in s2 (inverted)
                verts2_inv = self.__list_coords(s2, invert=True)

                # compare vertices and gather sets for s1 and s2
                # that are coincident
                s1_match_eh, s1_match_coords, s2_match_eh, s2_match_coords, \
                    match_dict = self.__get_matches(verts1, verts2_inv,
                                                    merge_tol)

                # matches were found, so continue
                if s1_match_eh != []:
                    # get only tris1 that have all match vertices
                    tris1 = self.__get_surf_triangles(s1_match_eh)

                    # get s2 tris to delete (no new surface needed)
                    tris2 = self.__get_surf_triangles(s2_match_eh)

                    # create new coincident surface
                    surf = self.mb.create_meshset()
                    self.mb.add_entities(surf, tris1)
                    self.mb.add_entities(surf, s1_match_eh)
                    tris_new = self.mb.get_entities_by_dimension(surf, 2)

                    # get skin of new merged surf (gets curve)
                    self.surf_curve[surf] = []
                    curve_verts = sk.find_skin(v1[1], tris1, True, False)
                    curve_edges = sk.find_skin(v1[1], tris1, False, False)

                    # if curve_verts/edges is empty, closed surf is created
                    # so no new curve is needed
                    if len(curve_verts) > 0:
                        # if not empty, make new curve
                        curve = self.mb.create_meshset()
                        self.mb.add_entities(curve, curve_verts)
                        self.mb.add_entities(curve, curve_edges)
                        self.surf_curve[s1].append(curve)
                        self.surf_curve[s2].append(curve)
                        self.surf_curve[surf].append(curve)

                        # remove merged verts and tris from each already
                        # existing surf
                        for vert_delete in s2_match_eh:

                            # get all triangles connected to the vert to be
                            # deleted
                            tris_adjust = self.mb.get_adjacencies(vert_delete,
                                                                  2, op_type=1)

                            # get the vert that will replace the deleted vert
                            replacement = match_dict[vert_delete]

                            # for every tri to be deleted, replace vert by
                            # setting connectivity
                            for tri in tris_adjust:
                                tri_verts = self.mb.get_connectivity(tri)
                                new_verts = [0, 0, 0]
                                for i, tv in enumerate(tri_verts):
                                    if tv == vert_delete:
                                        new_verts[i] = replacement
                                    else:
                                        new_verts[i] = tv

                                # set connectivity
                                self.mb.set_connectivity(tri, new_verts)

                    # remove from both sets (already in new surface)
                    self.mb.remove_entities(s1, tris1)
                    self.mb.remove_entities(s1, s1_match_eh)
                    self.mb.remove_entities(s2, tris2)
                    self.mb.remove_entities(s2, s2_match_eh)

                    # delete surf 2 (repeats)
                    self.mb.delete_entities(tris2)

                    # TAG INFORMATION

                    # assign sense tag to surface
                    # [forward=v1, backward=v2]
                    fwd = v1[1]
                    bwd = v2[1]
                    self.mb.tag_set_data(self.sense_tag, surf,
                                         [fwd, bwd])

                    # tag the new surface with the shared value
                    shared = \
                        list(set(self.isovol_meshsets[v1]['bounds']) &
                             set(self.isovol_meshsets[v2]['bounds']))
                    if not(bool(shared)):
                        print('no matching value!', v1, v2)
                        val = 0.0
                    else:
                        val = shared[0] * norm
                    self.mb.tag_set_data(self.val_tag, surf, val)

                    # add new surface to coincident surface list
                    match_surfs.append(surf)

                    # check if original surfaces are empty (no triangles)
                    # if so delete empty meshset and remove from list
                    s2_remaining = \
                        self.mb.get_entities_by_type(s2, types.MBTRI)
                    if len(s2_remaining) == 0:
                        # delete surface from list and mb instance
                        self.isovol_meshsets[v2]['surfs_EH'].remove(s2)

                    s1_remaining = \
                        self.mb.get_entities_by_type(s1, types.MBTRI)
                    if len(s1_remaining) == 0:
                        # delete from list and mb instance and move to
                        # next surf
                        self.isovol_meshsets[v1]['surfs_EH'].remove(s1)
                        break

        # After all comparisons have been made, add surfaces to lists
        self.isovol_meshsets[v1]['surfs_EH'].extend(match_surfs)
        self.isovol_meshsets[v2]['surfs_EH'].extend(match_surfs)
