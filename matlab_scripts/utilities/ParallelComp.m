classdef ParallelComp

    methods (Static)
        function parpool_init(varargin)
            poolobj = gcp('nocreate'); % If already pool, do not create new one.
            if isempty(poolobj)
                if ispc
                    parpool;
                else
                    if nargin == 0
                        parpool;
                    elseif nargin == 1
                        parpool(varargin{1});
                    elseif nargin == 2
                        parpool(varargin{1},varargin{2}+1);
                    end
                end
            end
            disp(poolobj)
        end
    end
end
