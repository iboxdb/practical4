require 'nn'

local ReQU = torch.class('nn.ReQU', 'nn.Module')

function ReQU:updateOutput(input)
  -- TODO
  self.output:resizeAs(input):copy(input) 
  local le = input:le( 0 )
  self.output[ le ] = 0 
  le =  le:eq( 0 )
  self.output:maskedCopy( le ,  input[le]:cmul(input[le]) )
  return self.output
end

function ReQU:updateGradInput(input, gradOutput)
  -- TODO
  self.gradInput:resizeAs(gradOutput):copy(gradOutput)
  local rinput = input:clone()   
  local le = rinput:le( 0 )
  rinput[ le ] = 0 
  le =  le:eq( 0 )
  rinput:maskedCopy( le ,  rinput[le]:mul(2) )
  self.gradInput:cmul(rinput)
  return self.gradInput
end

--require 'gnuplot'
--ii=torch.linspace(-5,5)

--m = nn.Square()
--m = nn.ReLU()
--m = nn.Sigmoid()
--m=nn.ReQU()

--oo=m:forward(ii)

--go=torch.ones(100)

--gi=m:backward(ii,go)
--gnuplot.plot({'f(x)',ii,oo,'+-'},{'df/dx',ii,gi,'+-'})
--gnuplot.grid(true) 
