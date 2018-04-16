"""This module only contains the Node class
"""

import numpy as np

class Node(object):
    """Three dimensional Node providing Dofs for displacements.

    Attributes
    ----------
    id : int or str
        Unique ID.
    reference_x : float
        Reference X coordinate.
    reference_y : float
        Reference Y coordinate.
    reference_z : float
        Reference Z coordinate.
    x : float
        Actual X coordinate.
    y : float
        Actual Y coordinate.
    z : float
        Actual Z coordinate.
    """

    def __init__(self, id, x, y, z):
        """Create a new node.

        Parameters
        ----------
        id : int or str
            Unique ID of the node.
        x : float
            Initial X coordinate of the node.
        y : float
            Initial Y coordinate of the node.
        z : float
            Initial Z coordinate of the node.
        """
        self.id = id
        self.x = x
        self.y = y
        self.z = z
        self.reference_x = x
        self.reference_y = y
        self.reference_z = z

    def GetReferenceLocation(self):
        """Location of the node in the reference configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the reference coordinates X, Y and Z.
        """
        x = self.reference_x
        y = self.reference_y
        z = self.reference_z

        return np.array([x, y, z], dtype=float)

    def GetActualLocation(self):
        """Location of the node in the actual configuration.

        Returns
        -------
        location : ndarray
            Numpy array containing the actual coordinates X, Y and Z.
        """
        x = self.x
        y = self.y
        z = self.z

        return np.array([x, y, z], dtype=float)

    def GetDisplacement(self):
        """Displacement of the node in the actual configuration.

        Returns
        -------
        displacement : ndarray
            A numpy array containing the displacements u, v and w.
        """
        return self.GetReferenceLocation() - self.GetActualLocation()

    def Update(self, dof_type, value):
        """FIXME"""

        if dof_type == 'u':
            self.x = self.reference_x + value
        elif dof_type == 'v':
            self.y = self.reference_y + value
        elif dof_type == 'w':
            self.z = self.reference_z + value
        else:
            raise RuntimeError('Node has no Dof of type {}'.format(dof_type))
