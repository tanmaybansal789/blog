+++
date = '2026-01-19T13:30:41Z'
draft = false
title = 'NLP: Part 2 - The Neural N-Gram Model'
+++

## Parameter Explosion

So far, we've looked at a model that uses probabilities of pairs of tokens for text generation.
However, we could extend this idea to triplets, quadruplets, and so on.

For example, consider a model that uses triplets (aptly named the *trigram model*).
A **trigram model** would have a **3D embedding tensor** of shape `(V, V, V)` (where `V` is the vocabulary size, one dimension for each successive token).
Using a vocabulary size of `65`, as seen before `65 ** 3 == 274,625` parameters, already a huge leap up from the previous `4225`.

In general, an `n`-gram model of vocabulary size `V` requires `V ** n` parameters, which, being **exponential**, does *not scale well at all*, and makes it essentially impossible to consider any reasonable context length.
It makes sense, though - at some point, you are basically storing the probability of **any given n-length sequence of tokens**, which is, of course, stupidly expensive.

## The Neural N-Gram

Instead, we could decide on a **2-dimensional embedding table**, where the width doesn't directly correlate to the vocabulary size, but instead, in some way, represents the word that it came from.
Then, stacking the vectors from the embedding lookup, we end up we some new vector of length `n_embed * (n - 1)` for a given `n`-gram. Then, using a simple **FFN** (*Feed Forward Network*) layer (`nn.Linear` in PyTorch terms), we can scale the `n_embed * n`-length vector into a `n_vocab` vector that can be softmaxxed to get the *exact same type of probability distribution`.

This is no longer a purely statistical n-gram model - by introducing this FFN layer, we allow the model to capture more complex relationships. 
*Only slightly, though - don't expect beautifully written English just yet...*

### Planning the model

Okay, so using this strategy, we require:
- an **embedding table**: `(n_vocab, n_embed)`
- a **weight matrix**: `(n_vocab, (n - 1) * n_embed)`
- a **bias vector**: `(n_vocab,)`

In the forward pass, we will:

1. **take the last `n - 1` tokens** from the token sequence.                      `(B, n - 1,)`
2. **look up each of them** in the **embedding table** and get the relevant rows  `(B, n - 1, n_embed)`
3. **concatenate the token embeddings** from oldest to newest.                    `(B, (n - 1) * n_embed,)`
4. **pass in this vector** to the **FFN** layer and get the output logits.        `(B, n_vocab)`

For token generation, we will:

5. **softmax** to get a probability distribution.                                 `(B, n_vocab)` (but now softmaxxed)
6. **sample** from it to get the next token, append to the existing context.      `(B, T + 1)`

## Implementation

Import the necessary libraries:
```python
import torch
from torch import nn
from torch.nn import functional as F
```

This time, we'll attempt to use a different device to the CPU (if possible).
Either way, it shouldn't make *too much* of a difference as long as you don't go crazy with the size of `n` in our n-gram.
```python
# device selection: prefer CUDA, then MPS (Apple Silicon), else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')
```

Implement the encoder, in the same way as in the bigram:
```python
class Encoder:
    def __init__(self, text):
        self.decoder = sorted(set(text))
        self.encoder = { c : i for i, c in enumerate(self.decoder) }

    def decode(self, l):
        return ''.join(self.decoder[i] for i in l)

    def encode(self, s):
        return [self.encoder[c] for c in s]

    @property
    def n_vocab(self):
        return len(self.decoder)
```

We will implement data preprocessing in the same way as before. The only difference is that we will only generate n-1 length training sequences and train on the following character rather than training on each pair of characters in a sequence.
```python
# split training/validation data    
def train_val_split(data, train_frac):
    i = int(len(data) * train_frac)
    return data[:i], data[i:]

# batch training data
def get_batch(data, n, batch_size):
    block_size = n - 1
    ix = torch.randint(len(data) - block_size, (batch_size,))
    xb = torch.stack([data[i : i + block_size] for i in ix]).to(device)
    yb = torch.stack([data[i + block_size] for i in ix]).to(device)
    return xb, yb
```

Let's define our model. It will be a little more complex than before, given that we store an embedding table, and a weight matrix/bias vector (which come packaged together in a PyTorch object, called `nn.Linear` - a linear transformation, one layer of a feedforward network.).
```python
class NGramModel(nn.Module):
    def __init__(self, vocab_size, n=3, embed_dim=32):
        super().__init__()
        self.n = n

        self.token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.fc = nn.Linear((n - 1) * embed_dim, vocab_size)
```
Let's walk through each attribute:
- `self.n`: the context window of the n-gram - given `n - 1` tokens, what is the following token?
- `self.token_embedding`: acts like an array of vectors, each one a **latent representation** of a specific token.
- `self.fc`: **(f)ully (c)onnected** layer projecting our concatenated token vectors for context into a vector representing an output distribution, for sampling - this is the guts of the next-token prediction engine.


Following this, let's write the function for computing the forward pass:
```python
    def forward(self, idx):
        B, T = idx.shape
        assert T == self.n - 1, "Input sequence length must be n-1"
        # embed tokens
        x = self.token_embedding(idx)     # (B, T, E)
        # concatenate context tokens
        x = x.view(B, T * x.size(-1))     # (B, (n-1)*E)
        logits = self.fc(x)               # (B, vocab)
        return logits
```
- `idx` is passed in, and the **(B)atch** and **(T)ime** dimensions extracted.
- We `assert` that the length of the time dimension equals the required context window - `self.n - 1`
- We embed each group of `n - 1` tokens in the batch into their own vectors.
- We concatenate each group of `n - 1` tokens so they can be passed into the forward layer.
- We pass this into the linear layer, which applies `x @ W.T + b` where `W` and `b` are our weight and bias respectively.
- We return our batch of logits.

Now, let's look at the autoregressive generation loop. It'll be similar to that for the bigram, except that the context we pass in is specifically the last `n - 1` tokens.
```python
    def generate(self, idx, n_toks=500):
        # assume that idx at least has n-1 tokens
        for _ in range(n_toks):
            # we will extract the last 'n - 1' tokens
            logits = self(idx[:, -self.n + 1:]) # (B, T)
            # softmax across the vocab dimension
            probs = F.softmax(logits, dim=1)
            # same as for bigram
            idx_next = torch.multinomial(probs, 1)
            # concatenate on the time dimension
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
```
After inputting `idx`, we apply the forward pass on the last `n - 1` tokens, use the functional `softmax` across the vocab dimension, sample the next batch tokens, and concatenate them. This process is repeated `n_toks` times.

Now, let's write the function which performs the training on the bigram. We will still be training in the same way - generate a batch, compute the predicted probabilities, evaluate with a loss function, perform gradient descent - the standard mini-batch loop.

```python
def train_ngram(model, train_data, batch_size=32, n_steps=10_000):
    optimiser = torch.optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss()

    for step in range(n_steps):
        xb, yb = get_batch(train_data, model.n, batch_size)

        logits = model(xb)
        loss = criterion(logits, yb)

        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        if step % 100 == 0:
            print(f'{step}: loss={loss.item()}')
```

Now, let's train on the same TinyShakespeare:
```python
if __name__ == '__main__':
    # load text and build encoder
    should_train = True

    input_path = 'p2_ngram_model/input.txt'
    model_path = 'p2_ngram_model/ngram_model.pt'

    text = open(input_path).read()
    encoder = Encoder(text)
    data = torch.tensor(encoder.encode(text), dtype=torch.long).to(device)
    train_data, val_data = train_val_split(data, 0.9)

    model = NGramModel(encoder.n_vocab, n=5).to(device)
    if should_train:
        train_ngram(model, train_data)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    print(encoder.decode((model.generate(torch.zeros((1, model.n - 1), dtype=torch.long, device=device), n_toks=1000).tolist())[0]))
```
```
MARENCEOTELIT:
In leve.

BEOMPreit?

LORDOUS:
Toserred by frangele,
Thon cures and ing, ard noad premaneselaresaro ghas, theppirhirences:
Dl, bees butwershof ont thargoof thry?
Whis chase wor hippers foutherwether farre breand and toree'llse, threpper.
Whan:
Why wore gathe thard ion aust:
Byour tfre astit sh htaclinge cone.

Hin fle kine
Furaelllakien yar fo murnes afonrellldastwe'ccle.

KING hewertor hat ankeno furge sige,
To to her :
If arss bane hat so fte thot, follesthe mane momwmy sof yfargund thes.

CUBEn:
I urseer wien hencengerer?

LOice, bore.

har makf bune isen tor why!

FLANUSAB
Linccenteprif thee anoure and tered's prat ice:
Thas that ftakis ond ot houl witht
Lor JgonI nor, mond
Sor hather will
Deatst pry d of'drtere, my comet,
Whisp Of hthe be ghen.

GLAUMBETHAMINGULE,
MENENO:
Lo gol wele, way to heaam ! CpIUKENO:
Her bond,
Belouid? nesprovin he anour and thes.

RAMEL:
For thee and tice.

DUKE VINCE:
I lo toof ous make alat:
Whatuthere sie;
Beverry'd way!
NGinfhere, Iy,
```
Okay, that is definitely more plausible than before -
- We even see some characters like `DUKE VINCE(NTIO)`, 
- Words like `for`, `thee`, `he`, `her`, `comet`, `ice`
- Along with a *slightly* more defined grammatical structure.

With 10 characters of context, the English generated feels a lot more convincing, but still utter garbage.
One improvement we could make is replacing our encoder with one that actually tokenises words, like those provided in `tiktoken`
I've also bumped up the training time to `90_000` - as the vocabulary is a lot larger, it's going to take a while for the loss to drop.
```python
if __name__ == '__main__':
    # load text and build encoder
    should_train = True

    input_path = 'p2_ngram_model/input.txt'
    model_path = 'p2_ngram_model/ngram_model_tokenised.pt'

    text = open(input_path).read()
    # encoder = Encoder(text)
    import tiktoken
    encoder = tiktoken.get_encoding('gpt2')

    data = torch.tensor(encoder.encode(text), dtype=torch.long).to(device)
    train_data, val_data = train_val_split(data, 0.9)

    model = NGramModel(encoder.n_vocab, n=5).to(device)
    if should_train:
        train_ngram(model, train_data, n_steps=90_000)
        torch.save({k: v.cpu() for k, v in model.state_dict().items()}, model_path)
    else:
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.to(device)
        model.eval()

    print(encoder.decode((model.generate(torch.zeros((1, model.n - 1), dtype=torch.long, device=device), n_toks=1000).tolist())[0]))
```
```
when, and soul
A thousander employ it, even put up on: impeach'd a while I back.

Second Servant:
RIVERS,
H OFmentER:
I cryach some vantage.

Clown:
Indeed, the state thou hastue;
Not like an lightning with my son,
 usual destiny.

First Citizen:
What shame merit I have themselves in Mantua?

NORTHUMBERLAND:
Saw puts thy comfort; and you have wholes content of my fellowship would have him brother'd by him.

MENENIUS:
Worthy temper, he is touch'd swift Gaunt,
Go to I the dispatch of roared and,
Say the mother, adieu,
And so few get your honest har unw shot spirit,
Thecher, hath a success did swim hands.
What Romeo are flies me to rise,
F late load upon my throne; at the rank Aufidius' which I saw the prince of hisLEONTES.

DUCHESS OF YORK:
It is a bawdy; that these sad last
Proclaim for Rome; take o'er, Lord of York
'Twere senators in a command's value!

CATESBY:
I beseech you to beaute
KING EDWARD IV:
You are welcome in a word.

AUTOLYCUS:
Now, my stay,
And gr confusionkind brMisrim my right heart:
But if it be so onlyied to my voice slew thy crown in all above all
The battle was made a piece ofar, surely, sir; for him, be aone;
Our prayers was about in heaven,--
avenuade me,
WARWICK:
The weary goodman so bad-like hath been desire that you was my dead.

ROMEO:
God swear new Margaret, sir, take the other of thy merchandiseits; not,--Sus kind Paris, try it had bound to our by step, two commanding forth

 point3 KING HENRY VI:
But if half he fell infection withal, nor thou fearful's garments?

LEONTES:
He would he beat him to beat him. Telly duk my prayers; be I for a servant.

HENRY BOLINGBROKE:
It shall captive upon her; discourse, but one gift;
Some choppingwith I have,
Minets haEainted seest it in st sense, silent consent and your do
Would I to have's suffer.
I was so look, to nature and age; children you well look
I know a health of state;
But yet, then warrant me to be VA cod hungry odd,
 warrant when I hither, church, some happy goodRel
Which mine being, whilst was not half such to add bear-made
Thyts, by Catesby, thou art
Liket his spirit to me:
Yetyour Rivers for me;
And I'lla,ae with you, their fittentry;
For one till well;
And shalt be at thy bwomen disdain'd:
The pleasure and k Richmond Edward'sEM.con evilsds,
And you to seek's which a sad Menaw Will deserves:
I do protest from 'able only brother day.

EXET:
Unto the part,
Now are,
As I did YORK:
Your fellow or greetings
As he but remedy:

Ex substitute o favver, who kiss his wit,
And can I in breathe
That Rome appear late honour 'tis time.

CORIOLANUS:
It
is not another, my woman and her;
In fire fatalere upon the carry,
Thatbe a slaughter- bad shadow of thy liege.

DUCHESS OF YORK:
 faces when he not come mouths swear
That I haveands
Have you like a cave: are
ont'd up
Have wont in high
bight devotion Caius Marcius.

BRUTUS:
It is the east with green, never should notBel unto the captaining of put found it.

DUCHESS OF YORK:
So will manners grant your grace!
The throat of them dead day.
A merit, sir?

ELBOW:
Confosedthee'er shroud to his shrew; attorneys bud truly notched that Richmond and no honour,
Into with hawthese post clamour puts up of those senators
That Harry'd-ains buzz'd dust taxMarry, a bawdyCan stop a sixteen whichman,

 veins haign
```
That's a clear improvement in legibility - nothing cohesive, of course.
We can also run it through some of the Linux kernel code, and get output that looks like this:
```c
!!!!: * +),
                struct sctphdr _tcph, *th;

                sh = skb_header_pointer(skb, offset, sizeof(_tcph), &_udph);
                        break;
                                goto out;
                        result = true;
                                offset += sprintf(temp+offset, "[%s] ", label);
                return 0;
}

/**
 * tomoyo_envp *envp)
{
        const struct common_audit_data *a)
{
        struct dentry *dentry = NULL;

        for (i = 0; i < TOMOYO_MODE_OWNER_EXECUTE:
                                value = stat->ino;
                memset(local_checked)) {
                if (!strcmp(word, tomoyo_argv *argv *start++ != '"' || *cp != '"')
                return NULL;
}

#endif /* TARGET_CORE_VERSION           "v5.0"

/*
         */
#define PR_REG_ISID_ID_LEN + 1];
        char revision[INQUIRY_BUF                               stat->gid = __bitmap.h>

/* List of "struct tomoyo_condition" entry.
 *
 * Returns true on success, false otherwise.
 *
 * @start: String to save.
 *
 * Copyright (C) 2005-2011  NTT DATA CORPORATION
 */

#include "common.h"
#include <linux/sctp.h>
#include <net/sock.h>
#include <linux/sctp.h>
#include <linux/gfp.h>
#include <net/af_unix.h>

/* List of "struct tomoyo_argv".
 * @ptr:  Pointer to "enum lockdown_reason level)
{
        char *cp = strchr(arg_ptr, argc, argv, argv++))
                        case TOMOYO_PATH1_MINOR:
                        case TOMOYO_MODE_OWNER_WRITE:
                if (offset > 0)
                return &obj->stat[stat_index];

                        switch (left) {
                state[len-1] = '\0'; /* Will restore later. */
        if (found) {
                        continue;
out:
                        *protocol_data;

        if (cp) {
                name = 0;
        struct scatterlist      *t_data_nents;
};

struct se_dev_entry *pr_reg_nacl;
        spinlock_t              lun_link;
        struct se_lun->lun_se_dev RCU read-side critical access */
        u64                     creation_time;
        bool            emulate_rest_reord;
        bool                    dentry = securityfs_create_file("lockdown", 0644, NULL, NULL,
                        /* Fetch values now. */
        u32                     dev_res_bin_isid;
        bool                    lun_entry_hlist;
        struct se_node_acl *fabric_stat_group;
};

struct se_node_acl *pr_res_key;
        u32             optimal_sectors;
        u64                     mapped_lun;
        struct list_head pr_reg_abort_list;
};

struct t10_alua {
        u64                             1024
#define SE_SENSE_KEY_OFFSET             7
#define SPC_SENSE_BUFFER                        96

enum target_prot_type;
        bool            emulate_write_cache;
        enum target_prot_type hw_pi_prot_type hw_pi_prot_type;
        int     tg_pt_gp_trans_delay_msecs;
        int                     lun_tg_pt_gp_lock;
        struct mutex            lun_acl;
        struct se_lun           xcopy_lun;
        struct work_struct      work;
```
The model has certainly learnt the C keywords and general structure of the code, as well as adopting some common names and patterns seen in the Kernel code. Compared to the Shakespeare example, this output appears significantly more convincing due to the very structured, repetitive nature of the programs.
