# Mapeamento heurístico de ameaças STRIDE por componente AWS
STRIDE_MAP = {
    "APIGateway": ["Spoofing", "Repudiation", "Denial of Service"],
    "Lambda": ["Elevation of Privilege", "Tampering"],
    "IAM": ["Spoofing", "Elevation of Privilege"],
    "S3": ["Information Disclosure", "Tampering"],
    "EC2": ["Elevation of Privilege", "Tampering", "DoS"],
    "Cognito": ["Spoofing", "Information Disclosure"],
    "RDS": ["Information Disclosure", "Tampering"],
    "Dynamo": ["Information Disclosure", "Tampering"],
    "SecretsManager": ["Information Disclosure", "Tampering"],
    "CloudWatch": ["Information Disclosure", "Repudiation"],
    "CloudTrail": ["Repudiation"],
    "KMS": ["Tampering", "Elevation of Privilege"],
    "StepFunctions": ["Elevation of Privilege"],
    "ALB_NLB": ["Spoofing", "DoS"],
    "SQS": ["Information Disclosure", "Tampering"],
    "SNS": ["Information Disclosure"],
    "Glue": ["Tampering"],
    "EMR": ["Tampering", "Elevation of Privilege"],
    "CloudFront": ["Information Disclosure", "Spoofing"],
    "AppSync": ["Spoofing", "Information Disclosure"],
    "CloudFormation": ["Tampering", "Elevation of Privilege"],
    "Route53": ["Spoofing", "Denial of Service"]
}

def analisar_ameacas_stride(componentes_detectados):
    """
    Recebe uma lista de nomes de componentes detectados e retorna um dicionário:
    { ameaca: [componentes...] }
    """
    from collections import defaultdict
    ameaças = defaultdict(list)

    for componente in componentes_detectados:
        for tipo in STRIDE_MAP.get(componente, []):
            ameaças[tipo].append(componente)

    return dict(ameaças)

