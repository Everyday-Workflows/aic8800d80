#!/usr/bin/env python3
"""Run the real fdrv RX reassembly code under AddressSanitizer.

The test extracts rwnx_rxdataind_aicwf(), rwnx_rx_data_skb(),
rwnx_rxdata_process_amsdu() and defrag_timeout_cb() from rwnx_rx.c and
compiles them in userspace with mocked sk_buffs, lists, locks and timers.
ASan reports use-after-free, double free, overflow and leaks; the mocked
skb_put() aborts like skb_over_panic(). Needs Python 3 and a C compiler with
ASan. No module is built, installed or loaded, and no hardware is used.
"""
from pathlib import Path
import os
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "drivers/aic8800/aic8800_fdrv/rwnx_rx.c"


def function(source, signature):
    # Brace counting is unreliable here: #ifdef branches open braces twice.
    # Kernel style closes a function with "}" alone at the start of a line.
    start = source.index(signature)
    end = source.index("\n}\n", start) + 3
    return source[start:end]


PRELUDE = r'''
#include <assert.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define KERNEL_VERSION(a, b, c) (((a) << 16) + ((b) << 8) + (c))
#define LINUX_VERSION_CODE KERNEL_VERSION(7, 3, 0)
#define AICWF_CFG80211_VERSION_CODE LINUX_VERSION_CODE
#define AICWF_RX_REORDER 1
#define CONFIG_RX_NETIF_RECV_SKB 1

typedef uint8_t u8, u8_l;
typedef uint16_t u16, u16_l;
typedef uint32_t u32;
typedef unsigned int gfp_t;
#define GFP_ATOMIC 1u
#define GFP_KERNEL 2u

#define printk(...) do { } while (0)
#define dev_err(dev, ...) do { (void)(dev); } while (0)
#define netdev_err(dev, ...) do { (void)(dev); } while (0)
#define AICWFDBG(level, ...) do { (void)(level); } while (0)
#define print_hex_dump(...) do { } while (0)
#define KERN_DEBUG ""
#define KERN_ERR ""
#define DUMP_PREFIX_NONE 0
#define LOGERROR 1
#define LOGINFO 4
#define LOGDEBUG 8
static int aicwf_dbg_level;
#define REG_SW_SET_PROFILING(hw, x) do { } while (0)
#define REG_SW_CLEAR_PROFILING(hw, x) do { } while (0)
#define unlikely(x) (x)
#define READ_ONCE(x) (x)
#define cmpxchg(p, o, n) ({ __typeof__(*(p)) _o = *(p); if (_o == (o)) *(p) = (n); _o; })
typedef struct { int counter; } atomic_t;
#define atomic_inc(a) ((a)->counter++)
#define atomic_xchg(a, v) ({ int _o = (a)->counter; (a)->counter = (v); _o; })
static unsigned long jiffies = 1000;
#define HZ 100
#define time_after(a, b) ((long)((b) - (a)) < 0)
#define msecs_to_jiffies(m) (m)
#define DEFRAG_MAX_WAIT 40
#define roundup(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define htons(x) __builtin_bswap16(x)
#define le32_to_cpu(x) (x)
#define container_of(p, type, member) ((type *)((char *)(p) - offsetof(type, member)))
#define MAC_ADDR_LEN 6
#define ETH_ALEN 6
#define ETH_P_802_3 0x0001
#define NET_XMIT_SUCCESS 0
#define NET_XMIT_DROP 1
#define RWNX_INVALID_STA 0xff
#define RWNX_AP_ISOLATE 1
#define NX_VIRT_DEV_MAX 4

struct list_head { struct list_head *next, *prev; };
static void INIT_LIST_HEAD(struct list_head *l) { l->next = l->prev = l; }
static int list_empty(const struct list_head *l) { return l->next == l; }
static void list_add_tail(struct list_head *n, struct list_head *h)
{ n->prev = h->prev; n->next = h; h->prev->next = n; h->prev = n; }
static void list_del_init(struct list_head *e)
{ e->prev->next = e->next; e->next->prev = e->prev; INIT_LIST_HEAD(e); }
#define list_for_each_entry(pos, head, member) \
    for (pos = container_of((head)->next, __typeof__(*pos), member); \
         &pos->member != (head); \
         pos = container_of(pos->member.next, __typeof__(*pos), member))
#define list_for_each_entry_safe(pos, n, head, member) \
    for (pos = container_of((head)->next, __typeof__(*pos), member), \
         n = container_of(pos->member.next, __typeof__(*pos), member); \
         &pos->member != (head); \
         pos = n, n = container_of(n->member.next, __typeof__(*n), member))

typedef struct { int held; } spinlock_t;
static void spin_lock_bh(spinlock_t *l) { assert(!l->held); l->held = 1; }
static void spin_unlock_bh(spinlock_t *l) { assert(l->held); l->held = 0; }
static void local_bh_disable(void) { }
static void local_bh_enable(void) { }

/* The RX path allocates with defrag_lock (a BH spinlock) held. */
static void check_atomic(gfp_t gfp)
{
    if (gfp != GFP_ATOMIC) {
        fprintf(stderr, "sleeping allocation in atomic context\n");
        abort();
    }
}
static void *kzalloc(size_t n, gfp_t gfp) { check_atomic(gfp); return calloc(1, n); }
static void kfree(void *p) { free(p); }

/* timers: tests fire or race them by hand */
struct timer_list { void (*fn)(struct timer_list *); int pending; };
static struct timer_list *last_timer;
static void timer_setup(struct timer_list *t, void (*fn)(struct timer_list *), int f)
{ (void)f; t->fn = fn; t->pending = 0; }
static int mod_timer(struct timer_list *t, unsigned long e)
{ (void)e; int was = t->pending; t->pending = 1; last_timer = t; return was; }
static int timer_delete(struct timer_list *t) { int was = t->pending; t->pending = 0; return was; }
static int timer_pending(const struct timer_list *t) { return t->pending; }
#define del_timer timer_delete
#define timer_container_of(var, t, member) container_of(t, __typeof__(*var), member)
static void fire_timer(struct timer_list *t) { t->pending = 0; t->fn(t); }

struct net_device { u8 dev_addr[6]; };
struct sk_buff {
    struct sk_buff *next, *prev;
    unsigned char *head, *data, *tail, *end;
    unsigned int len;
    struct net_device *dev;
    u16 protocol;
    u32 priority;
    char cb[48];
};
struct sk_buff_head { struct sk_buff *next, *prev; u32 qlen; };

static struct sk_buff *__dev_alloc_skb(unsigned int size, gfp_t gfp)
{
    check_atomic(gfp);
    struct sk_buff *skb = calloc(1, sizeof(*skb));
    skb->head = malloc(size + 64);
    skb->data = skb->tail = skb->head + 32;
    skb->end = skb->head + size + 64;
    return skb;
}
static void dev_kfree_skb(struct sk_buff *skb)
{
    if (!skb)
        return;
    free(skb->head);
    free(skb);
}
static unsigned int skb_tailroom(const struct sk_buff *s) { return s->end - s->tail; }
static unsigned int skb_headroom(const struct sk_buff *s) { return s->data - s->head; }
static unsigned char *skb_put(struct sk_buff *s, unsigned int n)
{
    unsigned char *t = s->tail;
    if (n > skb_tailroom(s)) {
        fprintf(stderr, "skb_over_panic: put %u tailroom %u\n", n, skb_tailroom(s));
        abort();
    }
    s->tail += n;
    s->len += n;
    return t;
}
static unsigned char *skb_pull(struct sk_buff *s, unsigned int n)
{ assert(n <= s->len); s->data += n; s->len -= n; return s->data; }
static unsigned char *skb_push(struct sk_buff *s, unsigned int n)
{ assert(n <= skb_headroom(s)); s->data -= n; s->len += n; return s->data; }
static void skb_reset_tail_pointer(struct sk_buff *s) { s->tail = s->data; }
static void skb_reset_mac_header(struct sk_buff *s) { (void)s; }
static void skb_reset_network_header(struct sk_buff *s) { (void)s; }
static struct sk_buff *skb_copy_expand(const struct sk_buff *s, int h, int t, gfp_t g)
{
    struct sk_buff *n = __dev_alloc_skb(s->len + h + t, g);
    skb_put(n, s->len);
    memcpy(n->data, s->data, s->len);
    return n;
}
static void __skb_queue_head_init(struct sk_buff_head *l)
{ l->next = l->prev = (struct sk_buff *)l; l->qlen = 0; }
static int skb_queue_empty(const struct sk_buff_head *l) { return l->qlen == 0; }
static struct sk_buff *skb_peek(const struct sk_buff_head *l) { return l->qlen ? l->next : NULL; }
static void __skb_queue_tail(struct sk_buff_head *l, struct sk_buff *s)
{ s->next = (struct sk_buff *)l; s->prev = l->prev; l->prev->next = s; l->prev = s; l->qlen++; }
static void __skb_queue_head(struct sk_buff_head *l, struct sk_buff *s)
{ s->prev = (struct sk_buff *)l; s->next = l->next; l->next->prev = s; l->next = s; l->qlen++; }
static struct sk_buff *__skb_dequeue(struct sk_buff_head *l)
{
    struct sk_buff *s = skb_peek(l);
    if (s) { s->next->prev = s->prev; s->prev->next = s->next; l->qlen--; }
    return s;
}

struct rwnx_txhdr { u8 hdr[64]; };
#define RWNX_SWTXHDR_ALIGN_SZ 4
struct ethhdr { u8 h_dest[6]; u8 h_source[6]; u16 h_proto; } __attribute__((packed));
static struct ethhdr *eth_hdr(const struct sk_buff *s) { return (struct ethhdr *)s->data; }
static int is_multicast_ether_addr(const u8 *a) { return a[0] & 1; }
struct ieee80211_hdr { u16 fc, dur; u8 addr1[6], addr2[6], addr3[6]; u16 seq; };

/* Frames handed to the stack; copied because the test frees them. */
static u8 delivered[8][4096];
static unsigned int delivered_len[8];
static int delivered_count;
static u16 eth_type_trans(struct sk_buff *s, struct net_device *d)
{
    (void)d;
    assert(delivered_count < 8 && s->len <= sizeof(delivered[0]));
    memcpy(delivered[delivered_count], s->data, s->len);
    delivered_len[delivered_count++] = s->len;
    return 0;
}
static void netif_receive_skb(struct sk_buff *s) { dev_kfree_skb(s); }
static int dev_queue_xmit(struct sk_buff *s) { dev_kfree_skb(s); return NET_XMIT_SUCCESS; }
static void rwnx_skb_align_8bytes(struct sk_buff *s) { (void)s; }

enum { NL80211_IFTYPE_STATION = 2, NL80211_IFTYPE_AP = 3, NL80211_IFTYPE_AP_VLAN = 4,
       NL80211_IFTYPE_MESH_POINT = 7, NL80211_IFTYPE_P2P_CLIENT = 8, NL80211_IFTYPE_P2P_GO = 9 };
enum { RX_STAT_FORWARD = 1 << 0, RX_STAT_DELETE = 1 << 2, RX_STAT_LEN_UPDATE = 1 << 3,
       RX_STAT_ETH_LEN_UPDATE = 1 << 4, RX_STAT_SPURIOUS = 1 << 6, RX_STAT_MONITOR = 1 << 7 };
enum { RWNX_RX_HD_DECR_UNENC, RWNX_RX_HD_DECR_WEP, RWNX_RX_HD_DECR_TKIP,
       RWNX_RX_HD_DECR_CCMP128, RWNX_RX_HD_DECR_CCMP256, RWNX_RX_HD_DECR_GCMP128,
       RWNX_RX_HD_DECR_GCMP256, RWNX_RX_HD_DECR_WAPI };

struct rx_vector_1 { int rssi1; };
struct rx_vector_2 { int x; };
struct hw_vect { u32 len; struct rx_vector_1 rx_vect1; struct rx_vector_2 rx_vect2; u8 decr_status; };
struct hw_rxhdr {
    struct hw_vect hwvect;
    u8 flags_is_amsdu, flags_is_80211_mpdu, flags_is_4addr, flags_need_reord, flags_upload;
    u8 flags_user_prio, flags_vif_idx, flags_sta_idx, flags_dst_idx, is_monitor_vif;
    u32 pad[4];
};
struct rxdesc_tag { u32 frame_len; };

struct net_device_stats { unsigned long rx_packets, rx_bytes, rx_dropped, tx_dropped, tx_errors; };
struct wireless_dev { int iftype; };
struct rwnx_vif {
    struct wireless_dev wdev;
    struct net_device *ndev;
    struct { int flags; struct list_head sta_list; } ap;
    struct net_device_stats net_stats;
    u8 vif_index;
    bool use_4addr, is_resending;
};
#define RWNX_VIF_TYPE(v) ((v)->wdev.iftype)
struct rwnx_sta { struct list_head list; u8 mac_addr[6]; bool valid; u8 vlan_idx; };
struct rwnx_hw {
    struct rwnx_vif *vif_table[NX_VIRT_DEV_MAX];
    struct rwnx_sta sta_table[8];
    spinlock_t defrag_lock, cb_lock;
    struct list_head defrag_list;
    struct { unsigned long amsdus_rx[64]; unsigned long last_rx; } stats;
    unsigned long fouraddr_said;
    atomic_t fouraddr_dropped;
    int monitor_vif;
    void *dev;
};
struct defrag_ctrl_info {
    struct list_head list;
    u8 sta_idx, tid;
    u16 sn;
    u8 next_fn;
    u16 frm_len;
    struct sk_buff *skb;
    struct timer_list defrag_timer;
    struct rwnx_hw *rwnx_hw;
};
struct aicwf_rx_priv { void *rwnx_vif; spinlock_t freeq_lock; struct list_head rxframes_freequeue; };
struct recv_msdu {
    struct sk_buff *first_fwd_skb, *last_fwd_skb, *first_resend_skb, *last_resend_skb;
    int is_ap_reord, ap_fwd_cnt, ap_resend_cnt;
};

static struct rwnx_vif *rwnx_rx_get_vif(struct rwnx_hw *hw, int idx)
{ return idx < NX_VIRT_DEV_MAX ? hw->vif_table[idx] : NULL; }
static void rwnx_rx_vector_convert(struct rwnx_hw *h, struct rx_vector_1 *a, struct rx_vector_2 *b)
{ (void)h; (void)a; (void)b; }
static u8 rwnx_rx_rtap_hdrlen(struct rx_vector_1 *v, bool f) { (void)v; (void)f; return 0; }
static int rwnx_rx_monitor(struct rwnx_hw *h, struct rwnx_vif *v, struct sk_buff *s,
                           struct hw_rxhdr *r, u8 l)
{ (void)h; (void)v; (void)s; (void)r; (void)l; abort(); }
static void cfg80211_rx_spurious_frame(struct net_device *d, const u8 *a, int l, gfp_t g)
{ (void)d; (void)a; (void)l; (void)g; }
static void cfg80211_rx_unexpected_4addr_frame(struct net_device *d, const u8 *a, int l, gfp_t g)
{ (void)d; (void)a; (void)l; (void)g; }
static void rwnx_rx_statistic(struct rwnx_hw *h, struct hw_rxhdr *r, struct rwnx_sta *s)
{ (void)h; (void)r; (void)s; }
static void remove_sec_hdr_mgmt_frame(struct hw_rxhdr *r, struct sk_buff *s) { (void)r; (void)s; abort(); }
static void rwnx_rx_mgmt_any(struct rwnx_hw *h, struct sk_buff *s, struct hw_rxhdr *r)
{ (void)h; (void)s; (void)r; abort(); }
static struct recv_msdu *reord_rxframe_alloc(spinlock_t *l, struct list_head *q)
{ (void)l; (void)q; abort(); }
static int reord_process_unit(struct recv_msdu *p, struct aicwf_rx_priv *r, struct sk_buff *s,
                              u16 sn, u8 tid, u8 fwd, u8 amsdu)
{ (void)p; (void)r; (void)s; (void)sn; (void)tid; (void)fwd; (void)amsdu; abort(); }
static int reord_flush_tid(struct aicwf_rx_priv *r, struct sk_buff *s, u8 tid)
{ (void)r; (void)s; (void)tid; return 0; }
static void rwnx_rx_data_skb_resend(struct rwnx_hw *h, struct rwnx_vif *v, struct sk_buff *s)
{ (void)h; (void)v; (void)s; abort(); }
static void rwnx_rx_data_skb_forward(struct rwnx_hw *h, struct rwnx_vif *v, struct sk_buff *s)
{ (void)h; (void)v; (void)s; abort(); }
'''

HARNESS = r'''
static struct rwnx_hw hw;
static struct rwnx_vif vif;
static struct net_device ndev;
static struct aicwf_rx_priv rx_priv;
static const u8 AP[6] = {0x02, 0xa0, 0, 0, 0, 1};
static const u8 ME[6] = {0x02, 0xbb, 0, 0, 0, 2};
static const u8 SA[6] = {0x02, 0xcc, 0, 0, 0, 3};

static void setup(void)
{
    memset(&hw, 0, sizeof(hw));
    INIT_LIST_HEAD(&hw.defrag_list);
    memset(&vif, 0, sizeof(vif));
    vif.wdev.iftype = NL80211_IFTYPE_STATION;
    vif.ndev = &ndev;
    INIT_LIST_HEAD(&vif.ap.sta_list);
    hw.vif_table[0] = &vif;
    delivered_count = 0;
    last_timer = NULL;
}

/* One USB RX buffer: hw_rxhdr, 4 bytes of USB padding, then a from-DS
 * 802.11 data header (QoS when qos_ctl >= 0) and the body.
 */
static struct sk_buff *rx_frame(int more, int fn, int qos_ctl, const u8 *body, unsigned int len)
{
    unsigned int hdr = qos_ctl >= 0 ? 26 : 24;
    struct sk_buff *skb = __dev_alloc_skb(sizeof(struct hw_rxhdr) + 4 + hdr + len, GFP_ATOMIC);
    skb->data = skb->tail = skb->head; /* rwnx_rxdataind_aicwf() runs at the buffer start */
    struct hw_rxhdr *r = (struct hw_rxhdr *)skb_put(skb, sizeof(*r));
    memset(r, 0, sizeof(*r));
    r->flags_upload = 1;
    r->flags_vif_idx = 0;
    r->flags_sta_idx = RWNX_INVALID_STA;
    r->flags_dst_idx = RWNX_INVALID_STA;
    r->hwvect.decr_status = RWNX_RX_HD_DECR_UNENC;
    memset(skb_put(skb, 4), 0, 4);
    u8 *h = skb_put(skb, hdr);
    memset(h, 0, hdr);
    h[0] = qos_ctl >= 0 ? 0x88 : 0x08;
    h[1] = 0x02 | (more ? 0x04 : 0);
    memcpy(h + 4, ME, 6);
    memcpy(h + 10, AP, 6);
    memcpy(h + 16, SA, 6);
    h[22] = 0x50 | fn; /* sequence number 5 */
    if (qos_ctl >= 0)
        h[24] = qos_ctl;
    memcpy(skb_put(skb, len), body, len);
    return skb;
}

static unsigned int llc_body(u8 *b, unsigned int payload, u8 fill)
{
    static const u8 llc[8] = {0xaa, 0xaa, 0x03, 0, 0, 0, 0x08, 0x00};
    memcpy(b, llc, 8);
    memset(b + 8, fill, payload);
    return 8 + payload;
}

static void rx(struct sk_buff *skb) { rwnx_rxdataind_aicwf(&hw, skb, &rx_priv); }

static void test_two_fragments(void)
{
    u8 b[64];
    setup();
    rx(rx_frame(1, 0, -1, b, llc_body(b, 20, 0x11)));
    assert(!list_empty(&hw.defrag_list) && delivered_count == 0);
    memset(b, 0x22, 30);
    rx(rx_frame(0, 1, -1, b, 30));
    assert(delivered_count == 1 && delivered_len[0] == 14 + 20 + 30);
    assert(!memcmp(delivered[0], ME, 6) && !memcmp(delivered[0] + 6, SA, 6));
    assert(delivered[0][12] == 0x08 && delivered[0][13] == 0x00);
    assert(delivered[0][14] == 0x11 && delivered[0][14 + 20] == 0x22);
    assert(list_empty(&hw.defrag_list));
    puts("two fragments: ok");
}

static void test_oversized_fragments(void)
{
    static u8 b[1200];
    setup();
    rx(rx_frame(1, 0, -1, b, llc_body(b, 1000, 0x33)));
    memset(b, 0x44, 1000);
    rx(rx_frame(1, 1, -1, b, 1000));
    rx(rx_frame(1, 2, -1, b, 1000)); /* exceeds the 2000-byte reassembly skb */
    assert(delivered_count == 0);
    struct timer_list *t = last_timer;
    assert(t && t->pending);
    fire_timer(t); /* drops the stale entry */
    assert(list_empty(&hw.defrag_list));
    puts("oversized fragments dropped: ok");
}

static void test_last_fragment_races_timer(void)
{
    u8 b[64];
    setup();
    rx(rx_frame(1, 0, -1, b, llc_body(b, 10, 0x55)));
    struct timer_list *t = last_timer;
    /* The timer expired and its callback is spinning on defrag_lock. */
    t->pending = 0;
    memset(b, 0x66, 10);
    rx(rx_frame(0, 1, -1, b, 10));
    assert(delivered_count == 1);
    t->fn(t); /* the callback now gets the lock and frees the entry */
    assert(list_empty(&hw.defrag_list));
    puts("last fragment racing timer: ok");
}

static void test_short_amsdu_subframe(void)
{
    u8 b[64] = {0};
    setup();
    /* A-MSDU subframe header whose length field (2) is below LLC/SNAP + type. */
    memcpy(b, ME, 6);
    memcpy(b + 6, SA, 6);
    b[12] = 0;
    b[13] = 2;
    rx(rx_frame(0, 0, 0x80, b, 40));
    assert(delivered_count == 0);
    puts("short A-MSDU subframe rejected: ok");
}

int main(void)
{
    test_two_fragments();
    test_oversized_fragments();
    test_last_fragment_races_timer();
    test_short_amsdu_subframe();
    return 0;
}
'''

FUNCTIONS = (
    "void rwnx_rxdata_process_amsdu(",
    "static bool rwnx_rx_data_skb(",
    "#if LINUX_VERSION_CODE < KERNEL_VERSION(4, 15, 0)\nvoid defrag_timeout_cb(ulong data)",
    "u8 rwnx_rxdataind_aicwf(",
)


def build_source(source):
    return PRELUDE + "\n".join(function(source, sig) for sig in FUNCTIONS) + HARNESS


class RxDefragTests(unittest.TestCase):
    def test_reassembly_under_asan(self):
        program = build_source(SOURCE.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            (tmp / "rx.c").write_text(program)
            binary = tmp / "rx"
            subprocess.run([os.environ.get("CC", "cc"), "-std=gnu11", "-g", "-O1",
                            "-fsanitize=address,undefined", "-fno-sanitize-recover=all",
                            "-fno-omit-frame-pointer", "-Wno-unused-function",
                            "-o", str(binary), str(tmp / "rx.c")], check=True)
            result = subprocess.run([str(binary)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
