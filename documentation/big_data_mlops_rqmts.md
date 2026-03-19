# Objetivo do Projeto
Desenvolver uma solução baseada em dados que resolva um problema real, aplicando
os conceitos e ferramentas de Big Data aprendidos em sala de aula. O projeto deve
obrigatoriamente abranger todas as etapas de um pipeline de dados, desde a coleta em
fontes diversas até a disponibilização dos insights gerados para análise.

# Entregáveis Big Data
- [ ] Fonte de dados com descrição
- [ ] Ingestão/Extração com descrição
- [ ] Transformação com descrição
- [ ] Carregamento com descrição
- [ ] Destino (onde ficará disponível para visualização) com descrição

* Primeira entrega *
- [ ] Documento de Arquitetura
    - [ ] Diagrama do pipeline de dados atual
    - [ ] Tecnologias já utilizadas e quais poderiam ser usadas para refinamento (tecnologias pagas) e justificativa da escolha
    - [ ] Arquitetura parcial implementada
    - [ ] Equipe responsável e divisão de tarefas
- [ ] Repositório no GitHub
    - [ ] Estrutura com pasta de: dados, src, documentação
    - [ ] README com: 
        - [ ] Nome e descrição do projeto;
        - [ ] Fonte dos dados;
        - [ ] Ferramentas já aplicadas.
    - [ ] Commits visíveis e com mensagens claras
- [ ] Demonstração Técnica (em aula)
    - [ ] Mostra do funcionamento da ingestão e/ou transformação com prints, outputs ou notebook.
    - [ ] Pode ser simulação parcial caso o pipeline ainda não esteja completo.
    - [ ] 8 min de apresentação
- [ ] Checklist Preenchido
    - Ingestão: ( ) Em progresso / ( ) Finalizado / ( ) Pendente
    - Armazenamento: ( ) Em progresso / ( ) Finalizado / ( ) Pendente
    - Transformação: ( ) Em progresso / ( ) Finalizado / ( ) Pendente

* Segunda Entrega *
- [ ] README
    - [ ] Introdução: Apresentação do tema e do problema.
    - [ ] Motivação: Relevância e justificativa da escolha do projeto.
    - [ ] Objetivo do Projeto: O que se pretende alcançar com a solução.
    - [ ] Metodologia (Pipeline de Dados): Descrição detalhada de cada etapa do pipeline (Fontes, Ingestão, Transformação, Carregamento, Destino), incluindo as tecnologias e a arquitetura da solução.
    - [ ] Resultados e Visualizações: Apresentação dos dashboards, gráficos e insights gerados.
    - [ ] Conclusões: Análise crítica dos resultados, dificuldades encontradas e trabalhos futuros.
- [ ] Pasta /src: Contendo todos os scripts, notebooks e códigos desenvolvidos.
- [ ] Pasta /notebooks: Jupyter Notebooks utilizados para exploração e análise.
- [ ] Pasta /dados (opcional): Amostras pequenas dos dados. Arquivos grandes não devem ser "commitados".
- [ ] Pasta /documentacao: Arquivos adicionais, como diagramas de arquitetura.

# Entregáveis MLOps
* Primeira Entrega*
- [ ] Pipeline de dados
    - [ ] arquivos .py
    - [ ] DVC
    - [ ] Pasta de data organizada
- [ ] Modelos
    - [ ] Treinar pelo menos 2 modelos 
    - [ ] Usar MLFlow
    - [ ] Integração com DVC
- [ ] Controle
    - [ ] DVC + Git
