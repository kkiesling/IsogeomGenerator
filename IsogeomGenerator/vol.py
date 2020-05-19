import sys
import os
import shutil
import warnings
import numpy as np
import math as m

import visit as v
from pymoab import core, types
from pymoab.rng import Range
from pymoab.skinner import Skinner


class IsoVolDatabase(object):
    """VisIt Step
    """

    def __init__(self, levels=None):
        # initialize levels list to be populated in various ways
        self.levels = levels

    def generate_volumes(self, filename, data,
                         dbname=os.getcwd()+"/tmp",
                         levels=None, levelfile=None, genmode=None, N=None,
                         minN=None, maxN=None):
        """Creates an STL file for each isovolume. N+1 files are
        generated and stored in the dbname folder.

        Input:
        ------
            filename: string, path to vtk file with the mesh
            data: string, name of the data whose values exist on the
                mesh (will be used to generate isocontours and
                isovolumes)
            dbname: (optional), string, name of folder to store created
                surface files. Must be absolute path!
                default: a folder called 'tmp' in the current directory

            level information: One of the following 3 sets of inputs must be
                provided if one of the level assignment methods was not called
                previously (assign_levels(), read_levels(), generate_levels()):
                Option 1: Assign levels
                    levels: list of floats, list of user-defined values to use
                        for contour levels
                Option 2: Read levels from file
                    levelfile: str, relative path to file with level
                        information. Each line of the file should have exactly
                        one float to be used as a level value.
                Option 3: Generate Levels
                    genmode: str, options are 'lin', 'log', or 'ratio'.
                        lin: N linearly spaced values between minN and maxN
                        log: N logarithmically spaced values between minN and
                            maxN
                        ratio: levels that are spaced by a constant ratio N.
                            minN will be used as minimum level value and the
                            maximum level value is less than or equal to maxN.
                    N: int or float, number of levels (int) to generate (lin
                        or log mode); or the ratio (float) to use to separate
                        levels (ratio mode).
                    minN: float, minimum level value
                    maxN: float, maximum level value
        """
        self.data = data
        self.db = dbname

        # gather level information
        if self.levels is None:
            # level information has not been previously assigned
            if levels is not None:
                self.assign_levels(levels)
            elif levelfile is not None:
                self.read_levels(levelfile)
            elif genmode is not None:
                if (N is None) or (minN is None) or (maxN is None):
                    raise RuntimeError("Generating levels requires input " +
                                       "values for N, minN, and maxN")
                else:
                    self.generate_levels(N, minN, maxN, mode=genmode)
            else:
                raise RuntimeError("Information for assigning levels " +
                                   "must be provided.")

        # Generate isovolumes using VisIT
        try:
            v.LaunchNowin()
        except:
            pass

        # create volumes
        v.OpenDatabase(filename)
        print("Generating isovolumes...")
        self.__generate_vols()
        print("...Isovolumes files generated!")
        v.CloseComputeEngine()

        # write levels to file in database
        self.__write_levels()

    def assign_levels(self, levels):
        """User defines the contour levels to be used as the isosurfaces.

        Input:
        ------
            levels: list of floats, list of user-defined values to use
                for contour levels
        """
        # make sure values are floats
        levels = [float(i) for i in levels]
        self.levels = sorted(levels)

    def read_levels(self, levelfile):
        """Read level values from a file. One value per line only.

        Input:
        ------
            levelfile: str, relative path to file with level information.
        """
        levels = []
        try:
            f = open(levelfile, 'r')
        except IOError:
            raise RuntimeError("Level file {} does not " +
                               "exist.".format(levelfile))

        levels = []
        lines = f.readlines()
        for line in lines:
            levels.append(float(line))

        self.levels = sorted(levels)

    def generate_levels(self, N, minN, maxN, mode='lin'):
        """Auto-generate evenly-spaced level values between the min and max
        value.

        Input:
        ------
            N: int or float, number of levels (int) to generate
                (lin or log mode); or the ratio (float) to use to separate
                levels (ratio mode).
            minN: float, minimum level value
            maxN: float, maximum level value
            mode: str, options are 'lin' (default), 'log', or 'ratio'.
                lin: N linearly spaced values between minN and maxN
                log: N logarithmically spaced values between minN and maxN
                ratio: levels that are spaced by a constant ratio N.
                    minN will be used as minimum level value and the maximum
                    level value is less than or equal to maxN.
        """
        if mode == 'lin':
            self.levels = list(np.linspace(minN, maxN,
                               num=N, endpoint=True))
        elif mode == 'log':
            base = 10.
            start = m.log(minN, base)
            stop = m.log(maxN, base)
            self.levels = list(np.logspace(start, stop, num=N,
                               endpoint=True, base=base))
        elif mode == 'ratio':
            # set minN as the minimum value and get all other values until maxN
            tmpmax = 0.
            self.levels = [minN]
            while tmpmax < maxN:
                next_val = self.levels[-1]*float(N)
                if next_val <= maxN:
                    self.levels.append(next_val)
                    tmpmax = next_val
                else:
                    break
        else:
            raise RuntimeError("Level generation mode {} not " +
                               "recognized.".format(mode))

    def __generate_vols(self):
        """Generates the isosurface volumes between the contour levels.
        Data files are exported as STLs and saved in the folder dbname.
        Files will be named based on their index corresponding to their
        level values (0.stl is lowest values).
        """
        # create folder to store data if it does not already exist
        i = 0
        while os.path.isdir(self.db):
            i += 1
            new_dir = self.db.rstrip("/").rstrip("-{}".format(i-1)) +\
                "-{}/".format(i)
            warnings.warn("Database {} exists. Using {} " +
                          "instead.".format(self.db, new_dir))
            self.db = new_dir
        os.makedirs(self.db + "/vols/")

        # plot the pseudocolor data in order to get volumes
        v.AddPlot("Pseudocolor", self.data)
        v.DrawPlots()

        # iterate over all isovolume levels
        for i, l in enumerate(self.levels):
            res = 0
            while res == 0:

                # lower bound
                if i == 0:
                    lbound = 0.0
                else:
                    lbound = self.levels[i-1]

                # upper bound
                ubound = self.levels[i]

                # get volume
                # res = 0 if no level found (should update to next level)
                res, ubound, skip_max = self.__get_isovol(lbound, ubound, i)
                if skip_max is True:
                    res = 1

        # get maximum isovolume level
        if not skip_max:
            lbound = self.levels[-1]
            ubound = 1.e200
            self.__get_isovol(lbound, ubound, i+1)

        # delete plots
        v.DeleteAllPlots()

    def __get_isovol(self, lbound, ubound, i):
        """Gets the volume selection for isovolume and export just the
        outer surface of the volume as STL.

        Input:
        ------
            lbound: float, lower boundary value for the isovolume
            ubound: float, upper boundary value for the isovolume
            i: int, surface number
        """
        # generate isovolume
        v.AddOperator("Isovolume")
        att = v.IsovolumeAttributes()
        att.lbound = lbound
        att.ubound = ubound
        v.SetOperatorOptions(att)

        # set operator setting to only get surfaces meshes
        v.AddOperator("ExternalSurface")

        # draw plot
        v.DrawPlots()

        # export current volume to folder
        e = v.ExportDBAttributes()
        e.dirname = self.db + "/vols/"
        e.db_type = "STL"
        e.filename = str(i)
        e.variables = self.data
        export_res = v.ExportDatabase(e)

        # check if exporting was successful or not and adjust values
        skip_max = False
        if export_res == 0:
            # export not successful because there was no data
            # get new upper bound
            warn_message = "Warning: no data to export between " \
                + "{} and {}.\n".format(lbound, ubound) \
                + "Increasing upper bound to next selected level."
            warnings.warn(warn_message)
            if ubound in self.levels:
                index = self.levels.index(ubound)
                ubound_old = ubound
                if ubound == max(self.levels):
                    skip_max = True
                else:
                    ubound = self.levels[index + 1]
                self.levels.remove(ubound_old)
            else:
                # it is the arbitrary upper level set and is not needed
                self.levels.remove(self.levels[-1])

        # delete the operators
        v.RemoveAllOperators()

        return export_res, ubound, skip_max

    def __write_levels(self):
        """Write the final level values used to a file that can be used by
        read_levels().
        """
        # store levels as a string
        level_str = ""
        for level in self.levels:
            level_str += str(level)
            level_str += "\n"

        # write to a file
        filepath = self.db + '/levelfile'
        with open(filepath, "w") as f:
            f.write(level_str)


class IsoSurfGeom(object):
    """MOAB step
    """

    def __init__(self):
        pass

    def create_geometry(self, tag_for_viz=False, norm=1.0, merge_tol=1e-5,
                        dbname=os.getcwd()+"/tmp", levelfile=None, tags=None,
                        sname=None, sdir=None):
        """Over-arching function to do all steps to create a single
        isosurface geometry for DAGMC using pyMOAB.

        Input:
        ------
            tag_for_viz: bool (optional), True to tag each triangle on
                every surface with the data value. Needed to visualize
                values in VisIt. Default=False.
            norm: float (optional), default=1. All data values will be
                multiplied by the normalization factor.
            merge_tol: float (option), default=1e-5 cm. Merge tolerance for
                mesh based merge of coincident surfaces. Recommended to be
                1/10th the mesh voxel size.
            dbname: (optional), string, name of folder to store created
                surface files. Must be absolute path.
                default: a folder called 'tmp' in the current directory
            levelfile: str (optional), relative path to file with level
                information. Each line of the file should have exactly
                one float to be used as a level value. If not provided, it
                will be looked for in the database folder (dbbame).
            tags: (optional), dict, set of names and values to tag on the
                geometry root set. Dictionary should be structured with each
                key as a tag name (str) and with a single value (float) for the
                tag. Example: {'NAME1':1.0, 'NAME2': 2.0}
            sname: (optional), str, name of file (including extension) for the
                written geometry file. Acceptable file types are VTK and H5M.
                Default name: isogeom.h5m
        """
        self.norm = norm
        self.merge_tol = merge_tol
        self.db = dbname

        # get level information
        if levelfile is not None:
            # read from provided file
            self.__read_levels(levelfile)
        else:
            # locate file in database
            filepath = self.db + '/levelfile'
            if os.path.exisits(filepath):
                self.__read_levels(filepath)
            else:
                raise RuntimeError("levelfile does not exist in database: " +
                                   "{}. ".format(filepath) +
                                   "Please provide levelfile location.")

        # Step 0: initialize meshsets
        self.mb = core.Core()
        self.isovol_meshsets = {}

        print("Reading database...")
        self.__read_database()
        print("... Reading complete!")

        # Step 1: Separate Isovolume Surfaces
        print("Separating isovolumes...")
        self.__separate_isovols()
        print("...Separation complete!")

        # Step 2: Merge Coincident Surfaces
        print("Merging surfaces...")
        self.__imprint_merge()
        print("...Merging complete!")

        # Step 3: Assign Parent-Child Relationship
        self.__make_family()

        if tag_for_viz:
            print('Tagging triangles with data...')
            self.__tag_for_viz()
            print('... tags complete')

        if tags is not None:
            self.__set_tags(tags)

        if sdir is None:
            sdir = self.dbname
        if sname is None:
            sname = 'isogeom.h5m'

        self.__write_geometry(sname, sdir)

    def __read_levels(self, levelfile):
        """Read level values from a file. One value per line only.

        Input:
        ------
            levelfile: str, relative path to file with level information.
        """
        levels = []
        f = open(levelfile, 'r')
        lines = f.readlines()
        for line in lines:
            levels.append(float(line))

        self.levels = sorted(levels)

    def __read_database(self):
        """Read the files from the database and initialize the meshset info.
        """
        for f in sorted(os.listdir(self.db + "/vols/")):
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
                    (self.levels[i-1], None)
            else:
                self.isovol_meshsets[iv_info]['bounds'] =\
                    (self.levels[i-1], self.levels[i])

    def __separate_isovols(self):
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

    def __get_surf_triangles(self, verts_good):
        """This function will take a set of vertice entity handles and
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

    def __get_matches(self, vertsA, vertsB):
        """Collects the set of entity handles and coordinates in set of
        vertsA and vertsB that match within the specified absolute
        tolerance (self.merge_tol).

        Input:
        ------
            vertsA: dictionary, key is the MOAB entity handle for the
                vertice and the value is a tuple of the coordinate
                (x, y, z)
            vertsB: dictionary, key is a tuple of the coordinate and the
                value is the MOAB entity handle for the coordinate

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
                tf = np.isclose(coord, bcoords, rtol=0, atol=self.merge_tol)

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

    def __compare_surfs(self, v1, v2):
        """finds coincident surfaces between two isovolumes.

        Input:
        ------
            v1/2: tuple, corresponds to the dictionary keys for two
                isovolumes in self.isovol_meshsets that will be compared
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
                    match_dict = self.__get_matches(verts1, verts2_inv)

                if s1_match_eh != []:
                    # matches were found, so continue

                    # get only tris1 that have all match vertices
                    tris1 = self.__get_surf_triangles(s1_match_eh)

                    # get s2 tris to delete (no new surface needed)
                    tris2 = self.__get_surf_triangles(s2_match_eh)

                    # create new coincident surface
                    surf = self.mb.create_meshset()
                    self.mb.add_entities(surf, tris1)
                    self.mb.add_entities(surf, s1_match_eh)
                    self.surf_curve[surf] = []

                    # get skin of new merged surf (gets curve)

                    curve_verts = sk.find_skin(surf, tris1, True, False)
                    curve_edges = sk.find_skin(surf, tris1, False, False)

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
                        list(set(self.isovol_meshsets[v1]['bounds'])
                             & set(self.isovol_meshsets[v2]['bounds']))
                    if not(bool(shared)):
                        print('no matching value!', v1, v2)
                        val = 0.0
                    else:
                        val = shared[0]*self.norm

                    self.mb.tag_set_data(self.val_tag, surf, val)

                    # add new surface to coincident surface list
                    match_surfs.append(surf)

                    # check if original surfaces are empty (no vertices)
                    # if so delete empty meshset and remove from list
                    s2_remaining = \
                        self.mb.get_entities_by_type(s2, types.MBVERTEX)
                    if len(s2_remaining) == 0:
                        # delete surface from list and mb instance
                        self.isovol_meshsets[v2]['surfs_EH'].remove(s2)

                    s1_remaining = \
                        self.mb.get_entities_by_type(s1, types.MBVERTEX)
                    if len(s1_remaining) == 0:
                        # delete from list and mb instance and move to
                        # next surf
                        self.isovol_meshsets[v1]['surfs_EH'].remove(s1)
                        break

        # After all comparisons have been made, add surfaces to lists
        self.isovol_meshsets[v1]['surfs_EH'].extend(match_surfs)
        self.isovol_meshsets[v2]['surfs_EH'].extend(match_surfs)

    def __imprint_merge(self):
        """Uses PyMOAB to check if surfaces are coincident. Creates a
        single surface where surfaces are coincident values are tagged
        on each surface. Surface senses are also determined and tagged.
        """

        # set up surface tag information (value and sense)
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

        # create dictionary of curves to match to surfaces:
        # key = surf eh, value = list of child curve eh
        self.surf_curve = {}

        # get list of all original isovolumes
        all_vols = sorted(self.isovol_meshsets.keys())
        for i, isovol in enumerate(all_vols):

            if i != len(self.levels):
                # do not need to check the last isovolume because it
                # will be checked against its neighbor already
                self.__compare_surfs(isovol, all_vols[i+1])

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

    def __make_family(self):
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
        curve_id = 0

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

                # tag surfaces
                self.mb.tag_set_data(geom_dim, surf_eh, 2)
                self.mb.tag_set_data(category, surf_eh, 'Surface')
                surf_id += 1
                self.mb.tag_set_data(global_id, surf_eh, surf_id)

        curve_id = 0
        for s in self.surf_curve.keys():
            for c in self.surf_curve[s]:
                # create relationship
                self.mb.add_parent_child(s, c)

                # tag curves
                self.mb.tag_set_data(geom_dim, c, 1)
                self.mb.tag_set_data(category, c, 'Curve')
                curve_id += 1
                self.mb.tag_set_data(global_id, c, curve_id)

    def __tag_for_viz(self):
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

    def __set_tags(self, tags):
        """Set provided tag values on the root set.

        Input:
        ------
            tags: dict, key=TAGNAME, value=TAGVALUE
        """
        rs = self.mb.get_root_set()
        for tagname, tagval in tags.items():
            tag = self.mb.tag_get_handle(tagname, size=1,
                                         tag_type=types.MB_TYPE_DOUBLE,
                                         storage_type=types.MB_TAG_SPARSE,
                                         create_if_missing=True)
            self.mb.tag_set_data(tag, rs, tagval)

    def __write_geometry(self, sname, sdir):
        """Writes out the geometry stored in memory.

        Input:
        ------
            sname: string, name of file to save written file
            sdir: string, absolute path for writing file
        """
        # check file extension of save name:
        ext = sname.split(".")[-1]
        if ext.lower() not in ['h5m', 'vtk']:
            print("WARNING: File extension {} ".format(ext) +
                  " not recognized. File will be saved as type .h5m.")
            sname = sname.split(".")[0] + ".h5m"
        # save the file
        save_location = sdir + "/" + sname
        self.mb.write_file(save_location)
        print("Geometry file written to {}.".format(save_location))
