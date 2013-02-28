classdef Embiggen
    %EMBIGGEN A class that makes Matrix-vector operations easier
    %
    %A magic class that allows one to do mixed matrix/vector operations by
    %virtually embiggening the vector to the size of the matrix. You can
    %do an operation between a matrix A and a vector b provided one of
    %their dimensions match, for example, you can write:
    %
    % C = A + Embiggen(b)
    %
    %Where A is a matrix (size 100x50)
    %      b be a vector (size 100x1)
    %
    %And the equation is understood as (in index notation)
    %                 C_ij = A_ij + b_j
    %
    % Which is more readable than the alternatives:
    %
    % C = A + b*ones(1,size(A,2)); 
    % C = A + repmat(b,1,size(A,2)); 
    % C = bsxfun(@plus,A,b); 
    %
    % How to use:
    %-----------
    % To construct a magic embiggener, use B = Embiggen(b). Here v is a 
    % vector of size mx1 or 1xn. Then use any one of the
    % following operators between an array A of size mxn and B:
    %
    %       A + Embiggen(b)
    %       A - Embiggen(b)
    %       A .* Embiggen(b)
    %       A ./ Embiggen(b)
    %       A .\ Embiggen(b)
    % 
    % And any one of the logical operations:
    %      
    %       A == Embiggen(b)
    %       A ~= Embiggen(b)
    %       A < Embiggen(b)
    %       A > Embiggen(b)
    %       A <= Embiggen(b)
    %       A >= Embiggen(b)
    %       A & Embiggen(b)
    %       A | Embiggen(b)
    %       xor(A, Embiggen(b))
    %
    % and any of the functions:
    %       max(A,Embiggen(b))
    %       min(A,Embiggen(b))
    %       rem(A,Embiggen(b))
    %       mod(A,Embiggen(b))
    %       atan2(A,Embiggen(b))
    %       hypot(A,Embiggen(b))
    %
    % More generally, Embiggen works between any two multidimensional
    % arrays A and B as long as each dimension of A and B is equal to each 
    % other, or equal to one. 
    %
    % Examples:
    %-----------
    % %Basic example:
    % A = randn(50,60);
    % b = randn(50,1);
    % c = randn(1,60);
    % A+Embiggen(b)
    % A+Embiggen(c)
    %
    % %Center and scale columns of a matrix
    % A = randn(50,60);
    % A = A - Embiggen(mean(A));
    % A = A ./ Embiggen(std(A));
    % mean(A) %vector of zeros
    % std(A) %vector of ones
    %
    % How it works: 
    %-----------
    % Calling Embiggen returns an instance of the Embiggen class, which
    % contains the sole property /data/. When an operator is used between
    % any value and an instance of Embiggen, the class method 
    % Embiggen.operatorname is called by matlab internally. bsxfun is used
    % to perform the actual operation.
    %
    % Known issues:
    %-----------
    % Embiggen is not an actual word:
    %   Please see http://en.wikipedia.org/wiki/Lisa_the_Iconoclast
    %
    % Information:
    %-----------
    % Author: Patrick Mineault
    %         patrick DOT mineault AT gmail DOT com
    %         http://xcorr.wordpress.com
    %
    % History: 13/05/2010: version 1.0
    % License: BSD
    %
    % SEE ALSO bsxfun    
    
    properties
        data
    end
    
    methods
        function [this] = Embiggen(data)
            this.data = data;
        end
        
        function [a] = plus(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@plus,arg1.data,arg2);
                else
                    a = bsxfun(@plus,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@plus,arg1,arg2.data);
            end
        end
        
        function [a] = minus(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@minus,arg1.data,arg2);
                else
                    a = bsxfun(@minus,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@minus,arg1,arg2.data);
            end
        end

        function [a] = times(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@times,arg1.data,arg2);
                else
                    a = bsxfun(@times,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@times,arg1,arg2.data);
            end
        end
        
        function [a] = power(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@power,arg1.data,arg2);
                else
                    a = bsxfun(@power,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@power,arg1,arg2.data);
            end
        end
        
        function [a] = rdivide(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@rdivide,arg1.data,arg2);
                else
                    a = bsxfun(@rdivide,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@rdivide,arg1,arg2.data);
            end
        end
        
        function [a] = ldivide(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@ldivide,arg1.data,arg2);
                else
                    a = bsxfun(@ldivide,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@ldivide,arg1,arg2.data);
            end
        end
        
        function [a] = min(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@min,arg1.data,arg2);
                else
                    a = bsxfun(@min,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@min,arg1,arg2.data);
            end
        end
        
        function [a] = max(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@max,arg1.data,arg2);
                else
                    a = bsxfun(@max,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@max,arg1,arg2.data);
            end
        end
        
        function [a] = rem(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@rem,arg1.data,arg2);
                else
                    a = bsxfun(@rem,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@rem,arg1,arg2.data);
            end
        end
        
        function [a] = mod(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@mod,arg1.data,arg2);
                else
                    a = bsxfun(@mod,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@mod,arg1,arg2.data);
            end
        end
        
        function [a] = atan2(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@atan2,arg1.data,arg2);
                else
                    a = bsxfun(@atan2,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@atan2,arg1,arg2.data);
            end
        end
        
        function [a] = hypot(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@hypot,arg1.data,arg2);
                else
                    a = bsxfun(@hypot,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@hypot,arg1,arg2.data);
            end
        end
        
        function [a] = eq(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@eq,arg1.data,arg2);
                else
                    a = bsxfun(@eq,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@eq,arg1,arg2.data);
            end
        end
        
        function [a] = ne(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@ne,arg1.data,arg2);
                else
                    a = bsxfun(@ne,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@ne,arg1,arg2.data);
            end
        end
        
        function [a] = lt(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@lt,arg1.data,arg2);
                else
                    a = bsxfun(@lt,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@lt,arg1,arg2.data);
            end
        end
        
        function [a] = le(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@le,arg1.data,arg2);
                else
                    a = bsxfun(@le,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@le,arg1,arg2.data);
            end
        end
        
        function [a] = gt(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@gt,arg1.data,arg2);
                else
                    a = bsxfun(@gt,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@gt,arg1,arg2.data);
            end
        end
        
        function [a] = ge(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@ge,arg1.data,arg2);
                else
                    a = bsxfun(@ge,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@ge,arg1,arg2.data);
            end
        end
        
        function [a] = and(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@and,arg1.data,arg2);
                else
                    a = bsxfun(@and,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@and,arg1,arg2.data);
            end
        end
        
        function [a] = or(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@or,arg1.data,arg2);
                else
                    a = bsxfun(@or,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@or,arg1,arg2.data);
            end
        end
        
        function [a] = xor(arg1,arg2)
            if strcmp(class(arg1),'Embiggen')
                if strcmp(class(arg2),'Embiggen')
                    a = bsxfun(@xor,arg1.data,arg2);
                else
                    a = bsxfun(@xor,arg1.data,arg2.data);
                end
            elseif strcmp(class(arg2),'Embiggen')
                a = bsxfun(@xor,arg1,arg2.data);
            end
        end

    end
    
end

