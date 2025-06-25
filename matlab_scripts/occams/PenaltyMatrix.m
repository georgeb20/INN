classdef PenaltyMatrix
    % PenaltyMatrix - A class to store the properties related to the penalty matrix for inversion.
    
    properties
        nnz = 0;                % Number of non-zero elements
        nrows = 0;              % Number of rows
        nrows_spatial = 0;      % Number of rows for R (spatial roughness should be inserted before e.g. anisotropy)
        
        colind = [];            % Column indices
        rowptr = [];            % Row pointers
        val = [];               % Values (RealPrec)
    end
    
    properties
        pen = [];               % Penalty matrix object (instance of PenaltyMatrix class)
    end
    
    methods
        function obj = PenaltyMatrix()
            % PenaltyMatrix constructor: Initializes the properties with default values.
        end
    end
end
