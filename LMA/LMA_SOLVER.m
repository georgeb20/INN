function rho_all = LMA_SOLVER(input_h5_file)

inch = 0.0254;
feet = 12 * inch;

freq = [2000 12000 48000]';
tool_sp = [408 954]';

freq_input = [2000 2000 12000 12000 48000 48000]';
tool_sp_input = [408 954 408 954 408 954]';

n_pixel = 30;
interface_lb = -70;
interface_ub = 70;
thick = interface_ub - interface_lb;
thick_delta = thick/n_pixel;
interface = linspace(interface_lb, interface_ub, n_pixel-1)';
plot_stair_tvd = linspace(interface_lb-thick_delta, interface_ub+thick_delta, n_pixel+1);
plot_grid_tvd = linspace(interface_lb-thick_delta/2, interface_ub+thick_delta/2, n_pixel);



rho_test = h5read(input_h5_file, '/rho');
disp(size(rho_test));
n_traces = size(rho_test,2);

rho_init = repelem([50,50,50], [12,9,9])';
epsilonr = repelem([1], [n_pixel])';


iter = 40;
Dip = 90;  % Angle of logging tool, 90 horizontal, 0 vertical
TVD = linspace(0, 0, n_traces);  % TVD of logging tool

perter = 0.01;
rho_all = zeros(n_pixel, n_traces);

for i_trace = 1:n_traces
    fprintf("On trace %d / %d\n", i_trace,n_traces);
    rho_truth = rho_test(:,i_trace);
    lambda = 5000;

    misfit_list = [];
    lambda_list = [];

    truth_curve_all = zeros(8*size(freq,1)*size(tool_sp,1), 1);
    for i_exp = 1:size(freq,1)*size(tool_sp,1)

        Resp_full_raw_truth = mexDipole(n_pixel, rho_truth, rho_truth, epsilonr, epsilonr, interface, TVD(i_trace), Dip , 0 ,...
            freq_input(i_exp), tool_sp_input(i_exp)/2 , -tool_sp_input(i_exp)/2);  % Forward solver
        H_field_tensor_truth = reshape(Resp_full_raw_truth,3,3);  % Reshape H tensor
        truth_curve = fromFieldtoCurves(H_field_tensor_truth);  % Calculate 8 curves base on H tensor
        truth_curve_all(1+(i_exp-1)*8:i_exp*8, 1) = truth_curve;  % record the 8 curves in each experiment to a list for all experiments
    end

    rho_temp = rho_init;
    temp_curve_all = zeros(8*size(freq,1)*size(tool_sp,1), 1);
    for i_exp = 1:size(freq,1)*size(tool_sp,1)
        Resp_full_raw_temp = mexDipole(n_pixel, rho_temp, rho_temp, epsilonr, epsilonr, interface, TVD(i_trace), Dip , 0 ,...
            freq_input(i_exp), tool_sp_input(i_exp)/2 , -tool_sp_input(i_exp)/2);
        H_field_tensor_temp = reshape(Resp_full_raw_temp,3,3);
        temp_curve = fromFieldtoCurves(H_field_tensor_temp);
        temp_curve_all(1+(i_exp-1)*8:i_exp*8, 1) = temp_curve;

        data_misfit_prev = norm(temp_curve_all - truth_curve_all);
    end


    for i_iter = 1:iter

        Jacob = zeros(8*size(freq,1)*size(tool_sp,1), n_pixel);
        % perter_matrix = exp(perter) * eye(n_pixel);
        perter_matrix = 10^perter * eye(n_pixel) + 1 - eye(n_pixel);
        for i_perter = 1:n_pixel
            % rho_perter = rho_temp + perter_matrix(:,i_perter);
            rho_perter = rho_temp .* perter_matrix(:,i_perter);

            pertur_curve_all = zeros(8*size(freq,1)*size(tool_sp,1), 1);
            for i_exp = 1:size(freq,1)*size(tool_sp,1)
                Resp_full_raw_pertur = mexDipole(n_pixel, rho_perter, rho_perter, epsilonr, epsilonr, interface, TVD(i_trace), Dip , 0 ,...
                    freq_input(i_exp), tool_sp_input(i_exp)/2 , -tool_sp_input(i_exp)/2);
                H_field_tensor_pertur = reshape(Resp_full_raw_pertur,3,3);
                pertur_curve = fromFieldtoCurves(H_field_tensor_pertur);
                pertur_curve_all(1+(i_exp-1)*8:i_exp*8, 1) = pertur_curve;
            end

            grad_gimj = (pertur_curve_all - temp_curve_all) / perter;
            Jacob(:, i_perter) = grad_gimj;
        end



        first_term = Jacob' * Jacob + lambda*eye(n_pixel);
        second_term = Jacob' * (truth_curve_all - temp_curve_all);
        delta_rho = first_term\second_term;  % in log10
        rho_curt = rho_temp .* 10.^(delta_rho);
        data_misfit_curt = Inf;
        while data_misfit_curt > data_misfit_prev

            temp_curve_all = zeros(8*size(freq,1)*size(tool_sp,1), 1);
            for i_exp = 1:size(freq,1)*size(tool_sp,1)
                Resp_full_raw_temp = mexDipole(n_pixel, rho_curt, rho_curt, epsilonr, epsilonr, interface, TVD(i_trace), Dip , 0 ,...
                    freq_input(i_exp), tool_sp_input(i_exp)/2 , -tool_sp_input(i_exp)/2);
                H_field_tensor_temp = reshape(Resp_full_raw_temp,3,3);
                temp_curve = fromFieldtoCurves(H_field_tensor_temp);
                temp_curve_all(1+(i_exp-1)*8:i_exp*8, 1) = temp_curve;
            end
            data_misfit_curt = norm(temp_curve_all - truth_curve_all);


            if data_misfit_curt <= data_misfit_prev
                %disp("Revious data misfit " + num2str(data_misfit_prev))
                %disp("Current data misfit " + num2str(data_misfit_curt))
                %disp("Misfit decrease, decrease lambda and go into next iteration.")
                lambda = lambda/5;
                % disp("Reduced lambda to " + num2str(lambda))
                misfit_list = [misfit_list, data_misfit_curt];
                lambda_list = [lambda_list, lambda];
            end


            if data_misfit_curt > data_misfit_prev
                %disp("Revious data misfit " + num2str(data_misfit_prev))
                %disp("Current data misfit " + num2str(data_misfit_curt))
                %disp("Misfit Iecrease! Increase lambda and retry rho update.")
                lambda = lambda*1.5;
                first_term_new = Jacob' * Jacob + lambda*eye(n_pixel);
                delta_rho = first_term_new\second_term;
                %disp("Increase lambda to " + num2str(lambda))
                % disp("Try rho update factor: ")
                % disp(exp(delta_rho)')
                rho_curt = rho_temp .* 10.^(delta_rho);
                misfit_list = [misfit_list, data_misfit_curt];
                lambda_list = [lambda_list, lambda];
            end
        end

        data_misfit_prev = data_misfit_curt;
        rho_temp = rho_curt;

    end

    rho_all(:, i_trace) = rho_temp;
end
end