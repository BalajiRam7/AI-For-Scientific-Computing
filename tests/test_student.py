'Checks for the completed neural-network starters.'
import sys
from pathlib import Path
import unittest
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'code')]
import burgers_reference as br
import starter_pinn as pinn
import starter_fno as fno

torch.set_num_threads(1)


class NeuralTests(unittest.TestCase):
    def test_residual_manufactured_function(self):
        class Exact(torch.nn.Module):
            def forward(self, x, t):
                return torch.exp(-t)*torch.sin(np.pi*x)
        torch.manual_seed(1)
        x = torch.rand(16,1,dtype=torch.float64,requires_grad=True)
        t = torch.rand(16,1,dtype=torch.float64,requires_grad=True)
        u = Exact()(x,t); nu = .03
        expected = (-1+nu*np.pi**2)*u + np.pi*torch.exp(-2*t)*torch.sin(np.pi*x)*torch.cos(np.pi*x)
        torch.testing.assert_close(pinn.pde_residual(Exact(), x,t,nu), expected)

    def test_pinn_backward_and_periodic_boundary(self):
        torch.manual_seed(0)
        model = pinn.MLP(8,2).double()
        batches = pinn.sample_points(16,8,8,'cpu',torch.float64)
        loss, parts = pinn.pinn_loss(model,*batches,br.NU_BENCHMARK)
        loss.backward()
        self.assertTrue(all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in model.parameters()))
        class Periodic(torch.nn.Module):
            def forward(self,x,t):
                return torch.sin(np.pi*x)*torch.exp(-t)
        _, parts = pinn.pinn_loss(Periodic(),*batches,.1)
        self.assertLess(parts['bc'], 1e-24)

    def test_periodic_loss_includes_derivative_term(self):
        class ValuePeriodicOnly(torch.nn.Module):
            def forward(self, x, t):
                return x**2*torch.exp(-t)
        torch.manual_seed(3)
        batches = pinn.sample_points(8,8,256,'cpu',torch.float64)
        _, parts = pinn.pinn_loss(ValuePeriodicOnly(),*batches,.1)
        t = batches[2][2]
        self.assertAlmostEqual(parts['bc'], float(torch.mean(16*torch.exp(-2*t))), places=10)

    def test_sampling_domains_and_initial_target(self):
        torch.manual_seed(5)
        (xc,tc,w), (xi,ti,u0), (xl,xr,tb) = pinn.sample_points(4096,4096,512,'cpu',torch.float64)
        for x in (xc.detach(), xi):
            self.assertTrue(bool(torch.all((x >= -1) & (x <= 1))))
            self.assertLess(float(x.min()), -.9); self.assertGreater(float(x.max()), .9)
        for t in (tc.detach(), tb):
            self.assertTrue(bool(torch.all((t >= 0) & (t <= 1))))
        self.assertTrue(bool(torch.all(ti == 0)))
        torch.testing.assert_close(u0, -torch.sin(np.pi*xi))
        self.assertTrue(bool(torch.all(xl == -1)) and bool(torch.all(xr == 1)))
        torch.testing.assert_close(w, torch.ones_like(w))

    def test_focused_sampling_distribution(self):
        torch.manual_seed(12)
        (x, t, w), _, _ = pinn.sample_points(100000,1,1,'cpu',sampling='focused')
        x = x.detach()
        self.assertTrue(bool(torch.all((x >= -1) & (x <= 1))))
        self.assertLess(abs(float((x.abs() <= .1).double().mean())-.55), .01)
        self.assertLess(abs(float(t.detach().mean())-.5), .01)
        self.assertEqual(tuple(w.shape), (100000,1))
        self.assertFalse(w.requires_grad)
        self.assertTrue(bool(torch.all(w > 0)))
        (_, _, w_off), _, _ = pinn.sample_points(1000,1,1,'cpu',sampling='focused',importance=False)
        torch.testing.assert_close(w_off, torch.ones_like(w_off))

    def test_spectral_identity_and_odd_grid(self):
        layer = fno.SpectralConv1d(1,1,6).double()
        with torch.no_grad():
            layer.weight.zero_(); layer.weight[...,0] = 1.
        n = 65; x = torch.arange(n,dtype=torch.float64)/n
        u = (torch.sin(2*np.pi*3*x)+.2*torch.cos(2*np.pi*2*x))[None,None,:]
        torch.testing.assert_close(layer(u),u,atol=1e-12,rtol=1e-12)
        with self.assertRaises(ValueError):
            layer(torch.zeros(1,1,8,dtype=torch.float64))

    def test_full_fno_is_not_low_pass(self):
        model = fno.FNO1d(modes=1,width=1,n_layers=1).double()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.lift.weight[0,0] = 1.
            model.pointwise[0].weight[0,0,0] = 1.
            model.project[0].weight[0,0] = 1.
            model.project[2].weight[0,0] = 1.
        x = torch.arange(128,dtype=torch.float64)/128
        u = torch.sin(2*np.pi*x)[None,:]
        out = model(u)
        spectrum = torch.fft.rfft(out,norm='forward')
        self.assertGreater(float(abs(spectrum[0,2]).detach()), .01)

    def test_no_gelu_after_last_fourier_block(self):
        model = fno.FNO1d(modes=1,width=1,n_layers=1).double()
        with torch.no_grad():
            for p in model.parameters(): p.zero_()
            model.lift.weight[0,0] = 1.
            model.pointwise[0].weight[0,0,0] = 1.
            model.project[0].weight[0,0] = 1.
            model.project[2].weight[0,0] = 1.
        u = torch.linspace(-2,2,16,dtype=torch.float64)[None,:]
        torch.testing.assert_close(model(u), torch.nn.functional.gelu(u))

    def test_fno_shapes_and_backward(self):
        torch.manual_seed(0); model = fno.FNO1d(6,4,2)
        for n in (32,64):
            y = model(torch.randn(3,n)); self.assertEqual(tuple(y.shape),(3,n))
        y.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.all(torch.isfinite(p.grad)) for p in model.parameters()))

    def test_train_epoch_runs(self):
        torch.manual_seed(4); model = fno.FNO1d(4,4,1)
        u = torch.randn(5,32); target = .5*u
        opt = torch.optim.Adam(model.parameters(),lr=.001)
        value = fno.train_epoch(model,u,target,opt,3)
        self.assertTrue(np.isfinite(value))


if __name__ == '__main__':
    unittest.main()
