
import inspect
import torch
try:
    import rfdetr
    from rfdetr import RFDETRMedium
    
    print("RFDETR version:", getattr(rfdetr, "__version__", "unknown"))
    print("RFDETR file:", rfdetr.__file__)
    
    print("\nRFDETRMedium.train signature:")
    print(inspect.signature(RFDETRMedium.train))
    
    print("\nRFDETRMedium.train docstring:")
    print(RFDETRMedium.train.__doc__)
    
    # Check if we can see source
    # print("\nRFDETRMedium.train source:")
    # print(inspect.getsource(RFDETRMedium.train))

except ImportError:
    print("rfdetr not installed")
except Exception as e:
    print(f"Error: {e}")
