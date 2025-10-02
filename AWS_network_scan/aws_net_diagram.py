#!/usr/bin/env python3
# Generates a network diagram per region: VPCs, subnets (public/private/isolated),
# Internet Gateways, NAT Gateways. Outputs: artifacts/aws_network_<region>.png

import argparse, os
from pathlib import Path
from collections import defaultdict
import boto3
from graphviz import Digraph

def gather(ec2):
    vpcs = ec2.describe_vpcs()["Vpcs"]
    subnets = ec2.describe_subnets()["Subnets"]
    rts = ec2.describe_route_tables()["RouteTables"]
    igws = ec2.describe_internet_gateways()["InternetGateways"]
    # NAT GW API is separate
    natgws = ec2.describe_nat_gateways()["NatGateways"] if hasattr(ec2, "describe_nat_gateways") else []

    # Map subnets → route table (explicit or main)
    vpc_main_rt = {}
    subnet_rt = {}
    for rt in rts:
        # remember the main route table per VPC
        if any(assoc.get("Main") for assoc in rt.get("Associations", [])):
            vpc_main_rt[rt["VpcId"]] = rt["RouteTableId"]
        # explicit subnet associations
        for assoc in rt.get("Associations", []):
            if "SubnetId" in assoc:
                subnet_rt[assoc["SubnetId"]] = rt["RouteTableId"]
    for sn in subnets:
        subnet_rt.setdefault(sn["SubnetId"], vpc_main_rt.get(sn["VpcId"]))

    # Classify subnets by default route
    rt_routes = {rt["RouteTableId"]: rt.get("Routes", []) for rt in rts}
    subnet_class = {}
    for sn in subnets:
        sn_id = sn["SubnetId"]
        rt_id = subnet_rt.get(sn_id)
        routes = rt_routes.get(rt_id, [])
        is_public = any(
            r.get("DestinationCidrBlock") == "0.0.0.0/0" and str(r.get("GatewayId", "")).startswith("igw-")
            for r in routes
        )
        via_nat = any(
            r.get("DestinationCidrBlock") == "0.0.0.0/0" and (
                str(r.get("NatGatewayId", "")).startswith("nat-") or str(r.get("NetworkInterfaceId","")).startswith("eni-")
            )
            for r in routes
        )
        if is_public:
            subnet_class[sn_id] = "public"
        elif via_nat:
            subnet_class[sn_id] = "private"
        else:
            subnet_class[sn_id] = "isolated"

    return vpcs, subnets, igws, natgws, subnet_rt, subnet_class, rts

def draw(region, vpcs, subnets, igws, natgws, subnet_rt, subnet_class, rts, outfile):
    g = Digraph("G", filename=outfile, format="png")
    g.attr(rankdir="LR", fontsize="10")

    subnets_by_vpc = defaultdict(list)
    for sn in subnets:
        subnets_by_vpc[sn["VpcId"]].append(sn)

    igw_by_vpc = defaultdict(list)
    for igw in igws:
        for att in igw.get("Attachments", []):
            if "VpcId" in att:
                igw_by_vpc[att["VpcId"]].append(igw)

    nat_by_subnet = defaultdict(list)
    for ng in natgws:
        snid = ng.get("SubnetId")
        if snid:
            nat_by_subnet[snid].append(ng)

    # Build per-VPC clusters
    for vpc in vpcs:
        vpc_id = vpc["VpcId"]
        with g.subgraph(name=f"cluster_{vpc_id}") as c:
            c.attr(label=f"{vpc_id} ({region})", style="rounded", color="lightgrey")

            # Internet Gateways
            for igw in igw_by_vpc.get(vpc_id, []):
                igw_id = igw["InternetGatewayId"]
                c.node(igw_id, label=f"IGW\n{igw_id}", shape="diamond")

            # Subnets + NAT GW nodes
            vpc_rts = {rt["RouteTableId"]: rt.get("Routes", []) for rt in rts if rt["VpcId"] == vpc_id}
            for sn in subnets_by_vpc.get(vpc_id, []):
                sn_id = sn["SubnetId"]
                az = sn.get("AvailabilityZone", "")
                name = next((t["Value"] for t in sn.get("Tags", []) if t.get("Key") == "Name"), "")
                role = subnet_class.get(sn_id, "?")
                fill = {"public": "#d4f4dd", "private": "#dbe9ff", "isolated": "#f4f4f4"}.get(role, "#ffffff")
                c.node(sn_id, label=f"{name or sn_id}\n{az}\n{role}", shape="box", style="filled,rounded", fillcolor=fill)

                # NATs inside subnets
                for ng in nat_by_subnet.get(sn_id, []):
                    nid = ng["NatGatewayId"]
                    c.node(nid, label=f"NAT\n{nid}", shape="triangle")
                    c.edge(sn_id, nid, label="0.0.0.0/0")

                # Default route edges to IGW/NAT if present
                rt_id = subnet_rt.get(sn_id)
                for r in vpc_rts.get(rt_id, []):
                    if r.get("DestinationCidrBlock") == "0.0.0.0/0":
                        if str(r.get("GatewayId","")).startswith("igw-"):
                            c.edge(sn_id, r["GatewayId"], label="0.0.0.0/0")
                        if str(r.get("NatGatewayId","")).startswith("nat-"):
                            c.edge(sn_id, r["NatGatewayId"], label="0.0.0.0/0")

    g.render(outfile, cleanup=True)

def main():
    ap = argparse.ArgumentParser(description="AWS network diagram generator")
    ap.add_argument("--regions", nargs="*", help="Regions to diagram, e.g. us-east-1 us-west-2")
    ap.add_argument("--profile", help="AWS profile name (optional)")
    ap.add_argument("--outfile", default="artifacts/aws_network", help="Output file prefix")
    args = ap.parse_args()

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    regions = args.regions or [session.region_name or "us-east-1"]

    Path("artifacts").mkdir(exist_ok=True)
    for region in regions:
        ec2 = session.client("ec2", region_name=region)
        vpcs, subnets, igws, natgws, subnet_rt, subnet_class, rts = gather(ec2)
        out = f"{args.outfile}_{region}"
        draw(region, vpcs, subnets, igws, natgws, subnet_rt, subnet_class, rts, out)
        print(f"✅ Wrote {out}.png")

if __name__ == "__main__":
    main()