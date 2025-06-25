classdef InversionIO

    methods (Static)
        function inversion_help() % to be modified
            fprintf(' \n');
            fprintf('Usage:  MARE2DEM [-F] [-J] [-scratch <scratchfolder>] <resistivity file> <output root name>\n');
            fprintf(' \n');
            fprintf('MARE2DEM has four optional flags (*normal inversion uses no flags):\n');
            fprintf(' \n');
            fprintf('      -F    Computes the forward response of the input model only.\n');
            fprintf('            The forward response is output to <resistivity file>.resp\n');
            fprintf(' \n');
            fprintf(' \n');
            fprintf('      -J    Outputs the full Jacobian matrix of the STARTING model for each iteration.\n');
            fprintf('            The Jacobian matrix is output to <resistivity file>.<iter#>.jacobianBin\n');
            fprintf('            Note: J is unweighted (i.e. it has NOT been normalized by the data uncertainties).\n');
            fprintf(' \n');
            fprintf('      -scratch <scratchfolder>  Use the specified directory for the scratch files\n');
            fprintf('            required for 2.5D CSEM inversion (but not MT). Optimally \n');
            fprintf('            this should be a local directory on each compute node and not\n');
            fprintf('            a networked directory.\n');
            fprintf(' \n');
            fprintf('MARE2DEM has one required parameter:\n');
            fprintf(' \n');
            fprintf('      <resistivity file> - This is the name of the input resistivity  \n');
            fprintf('      file. By convention, this file should have the extension .resistivity. \n');
            fprintf('      For example inputModel.0.resistivity.  The model found by each\n');
            fprintf('      inversion iteration is then output to a new resistivity file with the \n');
            fprintf('      iteration number incremented.  For example: inputModel.1.resistivity, \n');
            fprintf('      inputModel.2.resistivity, ... The corresponding model responses are \n');
            fprintf('      written to inputModel.1.resp, inputModel.2.resp,... \n');
            fprintf(' \n');
            fprintf('MARE2DEM has one optional parameter:\n');
            fprintf(' \n');
            fprintf('      <output file root> - With this option, the output files are \n');
            fprintf('      named <outputfileroot>.1.resistivity, <outputfileroot>.1.resp, \n');
            fprintf('      <outputfileroot>.2.resistivity, <outputfileroot>.2.resp,... \n');
            fprintf(' \n');

            exitINVERSION2D();
        end

        function displayBanner() % to be modified
            m2d_version = 'VERSION_NUMBER'; % Replace with the appropriate version number

            fprintf(' \n');
            fprintf('============================= MARE1DEM ===================================\n');
            fprintf(' \n');
            fprintf(' MARE1DEM: Modeling with Adaptively Refined Elements for 1.5D EM\n');
            fprintf('\n');
            fprintf(' %s\n', m2d_version);
            fprintf(' \n');
            fprintf(' A parallel goal-oriented adaptive finite element forward and inverse\n');
            fprintf(' modeling code for electromagnetic fields from electric dipoles, magnetic\n');
            fprintf(' dipoles and magnetotelluric sources in triaxial anisotropic conducting\n');
            fprintf(' media. Iterative adaptive mesh refinement is accomplished using the\n');
            fprintf(' goal-oriented error estimation method described in Key and Ovall (2011) \n');
            fprintf(' Inversion is accomplished with Occam''s method (Constable et al., 1987).\n');
            fprintf(' Key (2016) describes most of the features in the current version \n');
            fprintf(' of the code.\n');
            fprintf(' \n');
            fprintf(' When citing the code, please use the most recent reference:\n');
            fprintf(' \n');
            fprintf(' Key, K. MARE2DEM: a 2-D inversion code for controlled-source electromagnetic \n');
            fprintf('     and magnetotelluric data. Geophysical Journal International 207, \n');
            fprintf('     571–588 (2016).  \n');
            fprintf('\n');
            fprintf(' This work is currently supported by: \n');
            fprintf('\n');
            fprintf(' Electromagnetic Methods Research Consortium\n');
            fprintf(' Lamont-Doherty Earth Observatory\n');
            fprintf(' Columbia University\n');
            fprintf(' http://emrc.ldeo.columbia.edu\n');
            fprintf(' \n');
            fprintf(' Originally funded by:\n');
            fprintf('\n');
            fprintf(' Seafloor Electromagnetic Methods Consortium \n');
            fprintf(' Scripps Institution of Oceanography \n');
            fprintf(' University of California San Diego\n');
            fprintf('\n');
            fprintf(' Copyright (C) 2017-2021\n');
            fprintf(' Kerry Key\n');
            fprintf(' Lamont-Doherty Earth Observatory\n');
            fprintf(' Columbia University\n');
            fprintf(' http://emlab.ldeo.columbia.edu\n');
            fprintf(' \n');
            fprintf(' Copyright (C) 2008-2016\n');
            fprintf(' Kerry Key\n');
            fprintf(' Scripps Institution of Oceanography\n');
            fprintf(' University of California, San Diego\n');
            fprintf('\n');
            fprintf('\n');
            fprintf(' This file is part of MARE2DEM.\n');
            fprintf('\n');
            fprintf(' MARE2DEM is free software: you can redistribute it and/or modify\n');
            fprintf(' it under the terms of the GNU General Public License as published by\n');
            fprintf(' the Free Software Foundation, either version 3 of the License, or\n');
            fprintf(' (at your option) any later version.\n');
            fprintf('\n');
            fprintf(' MARE2DEM is distributed in the hope that it will be useful,\n');
            fprintf(' but WITHOUT ANY WARRANTY; without even the implied warranty of\n');
            fprintf(' MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the\n');
            fprintf(' GNU General Public License for more details.\n');
            fprintf('\n');
            fprintf(' You should have received a copy of the GNU General Public License\n');
            fprintf(' along with MARE2DEM. If not, see http://www.gnu.org/licenses/. \n');
            fprintf(' \n');
            fprintf('==========================================================================\n');
        end

    end
end
