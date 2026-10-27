from torchvision.transforms import GaussianBlur
import utils
import torch

class BaselineGenerator(object):
    def __init__(self, baseline):
        self.baseline = baseline
        self.baseline_ready = False

    def reset_baseline_status(self):
        self.baseline_ready = False

    def update_baseline_status(self, ready=True):
        self.baseline_ready = ready

    def _static_baseline_ready(self):
        if isinstance(self.baseline, str) and self.baseline.lower() == 'gaussian':
            return False    # Distributional baseline is dynamic
        if self.baseline_ready:
            return True     # Rests are static, even for explicand-dependent baseline
        return False

    def get_baseline(self):
        pass

class TextBaseline(BaselineGenerator):
    def __init__(self, baseline):
        assert isinstance(baseline, int), 'ONLY ALLOW CONSTANT BASELINE FOR "TEXT"'
        assert baseline >= 0, 'UNDEFINED BASELINE VALUE'
        super().__init__(baseline)

    def get_baseline(self, x):
        # self.update_baseline_status()
        return self.baseline    # Only support static baseline for texts atm
    
class ImageBaseline(BaselineGenerator):
    def __init__(self, baseline):
        super().__init__(baseline)
        self.baseline_value = None
        self.baseline_blur_kernel = GaussianBlur((31, 31), 5.0)

    def get_baseline(self, x, batch_size=None):
        """
        Different forms of the baseline specification is allowed, including:
        - [str]: current only support 'blur', which creates an explicand-dependent baseline
                 by passing it through a gaussian filter defined by `self.baseline_blur_kernel`
        - [int/float]:  a single constant value baseline that applies to all feature values
        - [tuple]:      a channel-specific constant value baseline, for example (0.485, 0.456, 0.406)
        - [tensor]:     a component-specific constant value baseline, should have the same shape as `x`
        Supported baseline types can be extended in the future
        """
        if self._static_baseline_ready():
            return self.baseline_value

        device = utils.get_device(x)
        if isinstance(self.baseline, str) and self.baseline.lower() == 'blur':   
            # explicand-specific baseline, requiring definition of `self.baseline_blur_kernel`, only apply to IMGs
            if self.baseline_blur_kernel is None: 
                print('Bluring kernel is not defined, use default')
                self.set_bluring_kernel()
            baseline_value = self.baseline_blur_kernel(x)
        elif isinstance(self.baseline, str) and self.baseline.lower() == 'gaussian':
            baseline_value = torch.randn((batch_size,)+x.shape[-3:])
        elif isinstance(self.baseline, (int, float)):
            # Constant value baseline, identical value across different channels (if applies)
            baseline_value = torch.tensor(self.baseline)
        elif isinstance(self.baseline, torch.Tensor):
            # Image-like baseline, having shape [C*H*W]
            baseline_value = self.baseline
        elif isinstance(self.baseline, tuple):
            # Constant value baseline for multi-channel image only
            baseline_value = torch.Tensor(self.baseline).reshape(len(self.baseline),1,1)
        else:
            assert 1 == 0, 'Unsupported baseline type'
        baseline_value = baseline_value.to(device, dtype=x.dtype)

        self._update_static_baseline(baseline_value)
        return baseline_value
    
    def _update_static_baseline(self, img_baseline):
        self.baseline_value = img_baseline
        self.update_baseline_status()

    def set_bluring_kernel(self, baseline_blur_kernel):
        """
        Parameters:
        -----------
        baseline_blur_kernel: blurring kernel for acquiring the baseline, 
                used only when specifying self.baseline='blur'
        """
        self.baseline_blur_kernel = baseline_blur_kernel 
