from . import Ansatz, KernelType


class KernelFactory:

    @staticmethod
    def create_pennylane_kernel(ansatz: Ansatz, measurement: str, type: KernelType):
        from ..core_implementation import PennylaneKernel
        return PennylaneKernel(ansatz,measurement,type)
    
    @staticmethod
    def create_qiskit_kernel(ansatz: Ansatz, measurement: str, type: KernelType):
        from ..core_implementation import QiskitKernel
        return QiskitKernel(ansatz,measurement,type)
    