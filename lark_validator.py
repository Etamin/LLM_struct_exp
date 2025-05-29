from lark import Lark


sql_grammar = '''
        program          : stmnt*
        stmnt            : select_stmnt | drop_stmnt
        select_stmnt     : select_clause from_clause? group_by_clause? having_clause? order_by_clause? limit_clause? SEMICOLON

        select_clause    : "select" selectables
        selectables      : column_name ("," column_name)*
        from_clause      : "from" source where_clause?
        where_clause     : "where" condition
        group_by_clause  : "group" "by" column_name ("," column_name)*
        having_clause    : "having" condition
        order_by_clause  : "order" "by" (column_name ("asc"|"desc")?)*
        limit_clause     : "limit" INTEGER_NUMBER ("offset" INTEGER_NUMBER)?

        // NOTE: there should be no on-clause on cross join and this will have to enforced post parse
        source           : joining? table_name table_alias?
        joining          : source join_modifier? JOIN source ON condition
        
        //source           : table_name table_alias? joined_source?
        //joined_source    : join_modifier? JOIN table_name table_alias? ON condition
        join_modifier    : "inner" | ("left" "outer"?) | ("right" "outer"?) | ("full" "outer"?) | "cross"
        
        condition        : or_clause+
        or_clause        : and_clause ("or" and_clause)*
        and_clause       : predicate ("and" predicate)*

        // NOTE: order of operator should be longest tokens first
   
        predicate        : comparison ( ( EQUAL | NOT_EQUAL ) comparison )* 
        comparison       : term ( ( LESS_EQUAL | GREATER_EQUAL | LESS | GREATER ) term )* 
        term             : factor ( ( "-" | "+" ) factor )*
        factor           : unary ( ( "/" | "*" ) unary )*
        unary            : ( "!" | "-" ) unary
                         | primary
        primary          : INTEGER_NUMBER | FLOAT_NUMBER | STRING | "true" | "false" | "null"
                         | IDENTIFIER

        drop_stmnt       : "drop" "table" table_name

        FLOAT_NUMBER     : INTEGER_NUMBER "." ("0".."9")*

        column_name      : IDENTIFIER
        table_name       : IDENTIFIER
        table_alias      : IDENTIFIER

        // keywords
        // define keywords as they have higher priority
        SELECT           : "select"
        FROM             : "from"
        WHERE            : "where"
        JOIN             : "join"
        ON               : "on"

        // operators
        STAR              : "*"
        LEFT_PAREN        : "("
        RIGHT_PAREN       : ")"
        LEFT_BRACKET      : "["
        RIGHT_BRACKET     : "]"
        DOT               : "."
        EQUAL             : "="
        LESS              : "<"
        GREATER           : ">"
        COMMA             : ","

        // 2-char ops
        LESS_EQUAL        : "<="
        GREATER_EQUAL     : ">="
        NOT_EQUAL         : ("<>" | "!=")

        SEMICOLON         : ";"

        IDENTIFIER       : ("_" | ("a".."z") | ("A".."Z"))* ("_" | ("a".."z") | ("A".."Z") | ("0".."9"))+

        %import common.ESCAPED_STRING   -> STRING
        %import common.SIGNED_NUMBER    -> INTEGER_NUMBER
        %import common.WS
        %ignore WS

        start: select_stmnt
'''

with open("Grammars/python.lark", "r") as f:
    grammar = f.read()
# l = Lark(sql_grammar)
l=Lark(grammar, parser='earley', start='start', ambiguity='explicit', propagate_positions=True, lexer='standard')
# print( l.parse("select a from t1") )
print( l.parse("import random\nimport math\n\ndef task_func(LETTERS=[chr(i) for i in range(97, 123)]):\n    result = {}\n\n    for letter in LETTERS:\n        num_integers = random.randint(1, 10)\n        random_integers = [random.randint(0, 100) for _ in range(num_integers)]\n        result[letter] = math.sqrt(sum((x - sum(random_integers) / len(random_integers)) ** 2 for x in random_integers) / len(random_integers))\n\n    return result") )