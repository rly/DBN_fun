%% RBM class definition
% written by Jon Berliner 4.7.1n3
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

		% TODO: assert this number is divisble into the number of instances in our dataset
		% NOTE: if we want to turn off batch mode, all we have to do is set this equal to the number of instances in our dataset
		nInstancesPerBatch;
        
	end

	methods
        
        %% constructor
        function obj = RBM(nVis, nHid,transFcn,CDn,lrate,leak, nInstancesPerBatch)
        % function obj = RBM(nVis, nHid,transFcn,CDn,lrate,leak)
            if(nargin > 0)
                obj.nVis = nVis;
                obj.nHid = nHid;
                obj.transFcn = transFcn;
                obj.CDn = CDn;
                obj.lrate = lrate;
                obj.leak = leak;
				obj.nInstancesPerBatch = nInstancesPerBatch;
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
        
		function batchIndices = createBatches(obj, data, nInstancesPerBatch)
			assert(mod(size(data, 1), nInstancesPerBatch) == 0); % nInstancesPerBatch should divide into the number of training instances
			nBatches = size(data, 1) / nInstancesPerBatch;
			batchIndices = nan(nBatches,nInstancesPerBatch);
			randOrder = randperm(size(data, 1));
			for i = 1:nBatches
				batchIndices(i,:) = randOrder((i-1)*nInstancesPerBatch+1:i*nInstancesPerBatch)
			end
		end

		function [SSE obs] = trainOnline(obj, data)
			% one at the end here indicates a single instance per batch
			return train(obj, data, 1);
		end

            
        %% online training
        function [SSE obj] = train(obj, data, nInstancesPerBatch)
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
            
			% get indices into the data to determine batch membership
			batchIndices = createBatches(obj, data, nInstancesPerBatch);
			nBatches = size(batchIndices, 1);

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
                    
                    for bi = 1:nBatches

						batch = data(batchIndices(bi,:),:);

                        if mod(bi,1000)==0
                            fprintf(['trial' num2str(bi) ' of ' num2str(nData) '\n']);
                        end

						for indToInput = 1:nInstancesPerBatch
							% input is now nInstancesPerBatch
							input = batch(indToInput,:); % get this training input from dataset
							
							visible = input; % init visible layer
						
							phidden0 = 1 ./ (1+ exp(-( visible*weightMat )) );
							hidrands = rand(1, nHid);
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

						end % end single training input

						% average the gradient across instances in the batch
						obj.weightMat = leak.*weightMat + lrate .* nanmean(weightGradient);
						
						% collect SSE for performance measuring
						edist = (input - recon).^2; % get Euclidean distance between input and reconstruction
						SSE(di) = sum(sum(edist));

				end % end batch training input
                    
                otherwise
                    error('not defined transfer function');
                    
            end % end switch
            
        end % end trainonline
        
    end % end methods
    
end
