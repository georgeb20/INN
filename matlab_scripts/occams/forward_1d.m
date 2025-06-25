function outval = forward_1d(x0 , model)
%FORWARD The forward function, calls the interface of forward calculation
%   Input:  VARIABLE X
%           MODEL DESCRIPTION
%   Output: TOOL RESPONSES


nL = length(x0); % number of layers

% read in the model parameters
ub = model.r_ub;
lb = model.r_lb;
m2 = x0(1:nL-model.nParamsAnis);

if model.anisotropy>0
    aniso_ratio = model.anisotropy;
elseif model.nParamsAnis
    anis_temp = x0(nL-model.nParamsAnis+1:nL);
    aniso_ratio = (model.anis_ub.*exp(anis_temp)+model.anis_lb.*exp(-anis_temp))./(exp(anis_temp)+exp(-anis_temp));
end
Rh = (ub*exp(m2)+lb*exp(-m2))./(exp(m2)+exp(-m2));
Rv = Rh*aniso_ratio;

Epsrh = ones(nL-model.nParamsAnis,1); %Used as comparison for simultaneous inversion of R and Epsr
Epsrv = Epsrh;


%% calculate the residual of data and forward response
freq = model.freq;
spac = model.spac;
n_freq = length(freq);
n_spac = length(spac);

%% read in the model parameters
Zbed = model.Zbed;  % location of boundaries
TVD = model.TVD;
Dip = model.Dip;
n_tvd = length(TVD);

spacing_all = repmat(spac , 1 , n_freq * n_tvd )';
freq_temp = repmat(freq,n_spac,1);
freq_temp = freq_temp(:);
freq_all = repmat(freq_temp,n_tvd,1);

%%
num_all = n_freq * n_spac * n_tvd;
n_freq_spacing = n_freq * n_spac;
num_resp = 8 * n_freq * n_spac;
curve_org = zeros(8,num_all);
% curve_org = zeros(n_freq * n_spac,8*n_tvd);

for i = 1:num_all
    i_tvd = floor((i-1)/(n_freq * n_spac)) + 1;
%     Resp_full_raw = mexDipole(nL-1, Rh, Rv, Epsrh, Epsrv, Zbed, TVD, Dip , 0 ,...
%         freq_all(i), 0 , -spacing_all(i));
    Resp_full_raw = mexDipole(nL-model.nParamsAnis, Rh, Rv, Epsrh, Epsrv, Zbed, TVD(i_tvd), Dip , 0 ,...
        freq_all(i), spacing_all(i)/2 , -spacing_all(i)/2);
    H_field_tensor = reshape(Resp_full_raw,3,3);
    temp = fromFieldtoCurves(H_field_tensor);
    temp = temp.';
    curve_org(:,i) = temp(:);
end

%% post process
Curve_mat = zeros(num_resp,length(TVD));
for i = 1:length(TVD)
        temp = curve_org(:,1+n_freq_spacing*(i-1):n_freq_spacing*i).';
        Curve_mat(:,i) = temp(:);
end
resp = Curve_mat(:);

% resp = mexGeosphere_full(nL-1, Rh, Rv, Epsrh, Epsrv, Zbed, TVD, Dip, model.freq, n_freq, model.spac, n_spac);

outval = resp;

end

