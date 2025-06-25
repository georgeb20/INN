function curves = fromFieldtoCurves(H_field_tensor)
Hfield_local = H_field_tensor.';

%% generate curves
Hxx = Hfield_local(1,1);
Hxz = Hfield_local(1,3);
Hyy = Hfield_local(2,2);
Hzx = Hfield_local(3,1);
Hzz = Hfield_local(3,3);
% Curves
att_anti = 20*log10(abs(((Hzz+Hzx)*(Hzz+Hxz))/((Hzz-Hzx)*(Hzz-Hxz))));
phs_anti = -180/pi*angle(((Hzz+Hzx)*(Hzz+Hxz))/((Hzz-Hzx)*(Hzz-Hxz)));
att_anis = -20*log10(abs(Hxx/Hyy));
phs_anis = 180/pi*angle(Hxx/Hyy);
att_bulk = -20*log10(abs((Hxx+Hyy)/(2*Hzz)));
phs_bulk = 180/pi*angle(-(Hxx+Hyy)/(2*Hzz));
att_symm = -20*log10(abs(((Hzz+Hzx)*(Hzz-Hxz))/((Hzz-Hzx)*(Hzz+Hxz))));
phs_symm = 180/pi*angle(((Hzz+Hzx)*(Hzz-Hxz))/((Hzz-Hzx)*(Hzz+Hxz)));
%
curves = zeros(8,1);
curves(1) = att_anti; curves(2) = phs_anti; curves(3) = att_anis; curves(4) = phs_anis;
curves(5) = att_bulk; curves(6) = phs_bulk; curves(7) = att_symm; curves(8) = phs_symm;

end