%% RBM class definition
% written by Jon Berliner 4.7.13
% comments beginning with ** are there to specifically teach about the learning algorithm
classdef RBM
    properties
        
		nVis;
		nHid;
        weightMat;
        
        transFcn;
        CDn;
        lrate;
        leak;
        
	end

	methods
        
        %% constructor
        function obj = RBM(nVis, nHid,transFcn,CDn,lrate,leak)
        % function obj = RBM(nVis, nHid,transFcn,CDn,lrate,leak)
            if(nargin > 0)
                obj.nVis = nVis;
                obj.nHid = nHid;
                obj.transFcn = transFcn;
                obj.CDn = CDn;
                obj.lrate = lrate;
                obj.leak = leak;
                % initialize all layer2layer weight matrices
                obj.weightMat = nan(nVis,nHid);
            end % end nargin
        end % end constructor
        
        
        %% initialize weights
        function obj = initializeWeights(obj,howSmall)
        % function obj = initializeWeights(howSmall)
            reinit = 'yes';
            % prompt to reinit if weights set at something besides all nans
            if sum(~isnan(obj.weightMat(:))) > 0 
                reinit = prompt('weights are set at something already!  are you sure you want to reinitialize?  type "yes" to continue');
            end
            
            if strcmp(reinit,'yes')
                obj.weightMat = howSmall.*rand(size(obj.weightMat));
            end
        end
        
            
        %% online training
        function [SSE obj] = trainonline(obj, data)
        % function [SSE obj] = trainonline(obj, data)
        % this function takes a matrix where every row is a single training sample
        % and trains in an online manner on the whole dataset.
        % it returns the updated RBM and a log of the SSE of Euclidean distance on each update
            
            
            % shorter names
            nHid = obj.nHid;
            nVis = obj.nVis;
            weightMat = obj.weightMat;
            transFcn = obj.transFcn;
            lrate = obj.lrate;
            leak = obj.leak;
            CDn = obj.CDn;
            
            % get params
            nData = size(data,1); % number of training inputs in dataset
            
            % init SSE table
            SSE = nan(1,nData);
            
            % init variables
            % ** only input, hidden0, recon, and hiddenN will be used for updating.  visible and *_temp named variables are used to arrive at recon and hiddenN.  *rands are there to stochastically fire layers
            phidden_temp = nan(1,nHid);
            hidden_temp = nan(1,nHid);

            phidden0 = nan(1,nHid);
            hidden0 = nan(1,nHid);

            phiddenN = nan(1,nHid);
            hiddenN = nan(1,nHid);

            visible = nan(1,nVis);

            precon = nan(1,nVis);
            recon = nan(1,nVis);

            hidrands = nan(1,nHid);
            visrands = nan(1,nVis);
            
            edist = nan;
            

            switch transFcn
                case 'sigmoid'
                    
                    for di = 1:nData
                        if mod(di,1000)==0
                            fprintf(['trial' num2str(di) ' of ' num2str(nData) '\n']);
                        end
                        input = data(di,:); % get this training input from dataset
                        
                        visible = input; % init visible layer
                    
                        phidden0 = 1 ./ (1+exp(-( visible*weightMat )) );
                        hidrands = rand(1,nHid);
                        hidden0 = double(phidden0 >= hidrands);
                        
                        phidden_temp = phidden0;
                        hidden_temp = hidden0;
                        
                        % pass activation down and up CDn times
                        count_cd = 1;
                        while count_cd <= CDn

                            % pass down
                            precon = 1 ./ (1+ exp(-( hidden_temp*weightMat' )) );
                            visrands = rand(1,nVis);
                            visible = double(precon >= visrands);
                            
                            % pass up
                            phidden_temp = 1 ./ (1+ exp(-( visible*weightMat )) ); % sigmoid-transform summed activation
                            hidrands = rand(1,nHid); % flip coins
                            hidden_temp = double(phidden_temp >= hidrands); % stochastically fire each hidden neuron with p(hidden)
                            
                            count_cd = count_cd+1;
                            
                        end % end CD
                        
                        % phiddenN = phidden; % can use phiddenN but using forced binary hiddenN for simplicity in learning about RBMs
                        hiddenN = hidden_temp;
                        recon = visible;

                        % update weights
                        % ** punish weights that led to a visible-hidden-layer coupling in the input that was not successfully conserved with the reconstruction
                        weightGradient = (input'*hidden0) - (recon'*hiddenN);
                        obj.weightMat = leak.*weightMat + lrate .* weightGradient;
                        
                        % collect SSE for performance measuring
                        edist = (input - recon).^2; % get Euclidean distance between input and reconstruction
                        SSE(di) = sum(sum(edist));
                        

                    end % end single training input
                    
                otherwise
                    error('not defined transfer function');
                    
            end % end switch
            
        end % end trainonline
        
    end % end methods
    
end